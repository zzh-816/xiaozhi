import os
import sys
import io
import time
import wave
from config.logger import setup_logging
from config.config_loader import get_project_dir
from core.providers.tts.base import TTSProviderBase

try:
    import sherpa_onnx
except ImportError:
    raise ImportError(
        "sherpa-onnx库未安装，请运行: pip install sherpa-onnx"
    )

TAG = __name__
logger = setup_logging()


# 捕获标准输出
class CaptureOutput:
    def __enter__(self):
        self._output = io.StringIO()
        self._original_stdout = sys.stdout
        sys.stdout = self._output

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self._original_stdout
        self.output = self._output.getvalue()
        self._output.close()

        # 将捕获到的内容通过 logger 输出
        if self.output:
            logger.bind(tag=TAG).debug(self.output.strip())


class TTSProvider(TTSProviderBase):
    def __init__(self, config, delete_audio_file):
        super().__init__(config, delete_audio_file)
        
        # 获取模型目录路径
        model_dir = config.get("model_dir", "models/sherpa-onnx-vits-zh-ll")
        
        # 处理相对路径，转换为绝对路径
        if not os.path.isabs(model_dir):
            project_dir = get_project_dir()
            self.model_dir = os.path.join(project_dir, model_dir)
        else:
            self.model_dir = model_dir
        
        # 检查模型目录是否存在
        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(f"模型目录不存在: {self.model_dir}")
        
        # 获取说话人ID，默认为0
        if config.get("private_voice"):
            self.speaker_id = int(config.get("private_voice"))
        else:
            self.speaker_id = int(config.get("sid", 0))
        
        # 获取音频格式，默认为wav
        self.audio_file_type = config.get("format", "wav")
        
        # 构建模型文件路径
        model_file = os.path.join(self.model_dir, "model.onnx")
        lexicon_file = os.path.join(self.model_dir, "lexicon.txt")
        tokens_file = os.path.join(self.model_dir, "tokens.txt")
        dict_dir = os.path.join(self.model_dir, "dict")
        
        # 检查必需的文件是否存在
        required_files = {
            "model.onnx": model_file,
            "lexicon.txt": lexicon_file,
            "tokens.txt": tokens_file,
        }
        
        for file_name, file_path in required_files.items():
            if not os.path.exists(file_path):
                raise FileNotFoundError(
                    f"模型文件不存在: {file_path}，请确保模型文件完整"
                )
        
        # 检查dict目录
        if not os.path.exists(dict_dir):
            logger.bind(tag=TAG).warning(
                f"字典目录不存在: {dict_dir}，将不使用字典目录"
            )
            dict_dir = ""
        
        # 获取规则FST文件（可选）
        rule_fsts = []
        rule_fst_files = [
            "number.fst",
            "phone.fst",
            "date.fst",
            "new_heteronym.fst",
        ]
        
        for fst_file in rule_fst_files:
            fst_path = os.path.join(self.model_dir, fst_file)
            if os.path.exists(fst_path):
                rule_fsts.append(fst_path)
        
        # 初始化TTS模型
        logger.bind(tag=TAG).info(f"正在加载sherpa-onnx-vits-zh-ll模型: {self.model_dir}")
        
        try:
            with CaptureOutput():
                # 构建OfflineTts配置
                # 注意：当前版本的sherpa-onnx不支持rule_fsts属性，因此不设置FST规则文件
                tts_config = sherpa_onnx.OfflineTtsConfig(
                    model=sherpa_onnx.OfflineTtsModelConfig(
                        vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                            model=model_file,
                            lexicon=lexicon_file,
                            tokens=tokens_file,
                            dict_dir=dict_dir if dict_dir else "",
                            data_dir="",  # 不使用data_dir
                            length_scale=1.0,
                            noise_scale=0.667,
                            noise_scale_w=0.8,
                        ),
                        num_threads=config.get("num_threads", 2),
                        debug=config.get("debug", False),
                        provider=config.get("provider", "cpu"),  # cpu 或 cuda
                    ),
                    max_num_sentences=config.get("max_num_sentences", 2),
                )
                
                self.tts = sherpa_onnx.OfflineTts(tts_config)
                
                # 记录FST文件信息（仅用于日志，不实际使用）
                if rule_fsts:
                    logger.bind(tag=TAG).debug(
                        f"检测到FST规则文件: {', '.join(os.path.basename(f) for f in rule_fsts)}（当前版本不支持）"
                    )
                logger.bind(tag=TAG).info(
                    f"sherpa-onnx-vits-zh-ll模型加载成功，说话人ID: {self.speaker_id}"
                )
        
        except Exception as e:
            logger.bind(tag=TAG).error(f"加载sherpa-onnx-vits-zh-ll模型失败: {e}")
            raise
    
    async def text_to_speak(self, text, output_file):
        """
        将文本转换为语音
        
        Args:
            text: 要合成的文本
            output_file: 输出文件路径，如果为None则返回音频字节数据
        
        Returns:
            如果output_file为None，返回音频字节数据；否则返回None
        """
        try:
            # 记录TTS开始时间
            tts_start_time = time.time()
            
            # 使用sherpa-onnx进行语音合成
            audio = self.tts.generate(text, sid=self.speaker_id, speed=1.0)
            
            # 记录TTS首次响应时间（生成完成即首次响应）
            tts_first_response_time = time.time()
            first_response_latency = tts_first_response_time - tts_start_time
            logger.bind(tag=TAG).info(f"TTS首次响应耗时: {first_response_latency:.3f}s")
            
            # 获取音频数据
            import numpy as np
            
            # audio.samples 可能是列表或numpy数组，统一转换为numpy数组
            samples = audio.samples
            if isinstance(samples, list):
                samples = np.array(samples, dtype=np.float32)
            else:
                samples = np.asarray(samples, dtype=np.float32)
            
            sample_rate = audio.sample_rate  # 采样率
            
            # 确保数据是一维数组
            if len(samples.shape) > 1:
                samples = samples.flatten()
            
            # 将[-1, 1]范围的float32转换为[-32768, 32767]范围的int16
            # 使用clip确保值在有效范围内
            samples_int16 = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
            
            # 创建WAV格式的字节数据
            wav_bytes = io.BytesIO()
            with wave.open(wav_bytes, "wb") as wav_file:
                wav_file.setnchannels(1)  # 单声道
                wav_file.setsampwidth(2)  # 16位采样
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(samples_int16.tobytes())
            
            # 获取WAV字节数据
            wav_bytes.seek(0)  # 确保指针在开始位置
            wav_data = wav_bytes.getvalue()
            
            # 保存到文件或返回字节数据
            if output_file:
                # 确保输出目录存在
                os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
                with open(output_file, "wb") as f:
                    f.write(wav_data)
                logger.bind(tag=TAG).debug(f"音频文件已保存: {output_file}")
            else:
                return wav_data
        
        except Exception as e:
            error_msg = f"sherpa-onnx-vits-zh-ll TTS合成失败: {e}"
            logger.bind(tag=TAG).error(error_msg)
            raise Exception(error_msg)

