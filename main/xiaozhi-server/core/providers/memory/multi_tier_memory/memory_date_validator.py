"""记忆日期验证和纠正模块"""
import re
from datetime import datetime, timedelta
from typing import Tuple, Optional

from ..base import logger

TAG = __name__

# 相对时间词模式（过去到上周，未来到下下周）
RELATIVE_TIME_PATTERNS = [
    # 基础相对时间
    r'今天', r'昨天', r'明天', r'后天', r'大后天',
    r'前天', r'大前天',
    # 相对周词
    r'这周[一二三四五六日天]',
    r'下周[一二三四五六日天]',
    r'下周$',  # 单独的"下周"（没有具体星期几）
    r'下下[一二三四五六日天]',
    r'本周[一二三四五六日天]',
    r'本[一二三四五六日天]',
    r'上周[一二三四五六日天]',
    # 相对天数
    r'\d+天后', r'\d+天前',
    r'过\d+天', r'\d+天之后',
    # 其他相对时间
    r'最近', r'前几天', r'过几天',
    # 相对月份
    r'下个月',
]


def has_time_information(memory_text: str) -> bool:
    """
    检测记忆文本是否包含相对时间词
    
    Args:
        memory_text: 记忆文本，格式如 "2026-01-07: 用户下周三有英语考试"
    
    Returns:
        bool: 是否包含相对时间词
    """
    # 1. 必须有日期格式
    date_match = re.match(r'^\d{4}-\d{2}-\d{2}:\s*', memory_text)
    if not date_match:
        return False
    
    # 2. 提取内容部分
    content = memory_text[date_match.end():]
    
    # 3. 检测是否包含相对时间词
    for pattern in RELATIVE_TIME_PATTERNS:
        if re.search(pattern, content):
            return True
    
    return False


def detect_relative_time_word(content: str) -> Optional[str]:
    """
    检测内容中的相对时间词
    
    Args:
        content: 内容文本
    
    Returns:
        Optional[str]: 检测到的相对时间词，如果没有则返回None
    """
    # 按优先级检测（更具体的模式先检测）
    priority_patterns = [
        (r'下下[一二三四五六日天]', '下下周'),
        (r'下周[一二三四五六日天]', '下周'),  # 先检测"下周X"（有具体星期）
        (r'下周(?![\u4e00-\u9fa5])', '下周'),  # 再检测单独的"下周"（后面不是汉字，即没有具体星期，默认下周一）
        (r'这周[一二三四五六日天]', '这周'),
        (r'本周[一二三四五六日天]', '本周'),
        (r'本[一二三四五六日天]', '本周'),
        (r'上周[一二三四五六日天]', '上周'),
        (r'大后天', '大后天'),
        (r'后天', '后天'),
        (r'明天', '明天'),
        (r'今天', '今天'),
        (r'大前天', '大前天'),
        (r'前天', '前天'),
        (r'昨天', '昨天'),
        (r'\d+天后', 'N天后'),
        (r'\d+天前', 'N天前'),
        (r'过\d+天', '过N天'),
        (r'\d+天之后', 'N天之后'),
        (r'下个月', '下个月'),
        (r'最近', '最近'),
        (r'前几天', '前几天'),
        (r'过几天', '过几天'),
    ]
    
    for pattern, time_type in priority_patterns:
        match = re.search(pattern, content)
        if match:
            matched_text = match.group(0)
            # 特殊处理：如果匹配到"下周"但后面跟着星期几的汉字，跳过（应该匹配"下周X"）
            if matched_text == '下周':
                # 检查"下周"后面是否跟着星期几的汉字
                match_pos = match.end()
                if match_pos < len(content):
                    next_char = content[match_pos]
                    if next_char in '一二三四五六日天':
                        continue  # 跳过，应该匹配"下周X"
            return matched_text  # 返回匹配到的完整文本
    
    return None


def _weekday_to_num(weekday: str) -> int:
    """
    将星期几转换为数字（0=周一，6=周日）
    
    Args:
        weekday: 星期几，如"星期一"、"周二"
    
    Returns:
        int: 星期几的数字表示
    """
    weekday_map = {
        '星期一': 0, '周一': 0, '一': 0,
        '星期二': 1, '周二': 1, '二': 1,
        '星期三': 2, '周三': 2, '三': 2,
        '星期四': 3, '周四': 3, '四': 3,
        '星期五': 4, '周五': 4, '五': 4,
        '星期六': 5, '周六': 5, '六': 5,
        '星期日': 6, '周日': 6, '星期天': 6, '天': 6,
    }
    return weekday_map.get(weekday, 0)


def _char_to_weekday_num(char: str) -> int:
    """
    将汉字星期转换为数字
    
    Args:
        char: 汉字，如"一"、"二"、"日"
    
    Returns:
        int: 星期几的数字表示
    """
    char_map = {'一': 0, '二': 1, '三': 2, '四': 3, '五': 4, '六': 5, '日': 6, '天': 6}
    return char_map.get(char, 0)


def calculate_correct_date(relative_time: str, current_date: str, current_weekday: str) -> Optional[datetime]:
    """
    根据相对时间词计算正确日期
    
    Args:
        relative_time: 相对时间词，如"下周三"、"明天"、"3天后"
        current_date: 当前日期，格式 "YYYY-MM-DD"
        current_weekday: 当前星期几，如"星期一"
    
    Returns:
        Optional[datetime]: 计算出的正确日期，如果无法计算则返回None
    """
    try:
        current_dt = datetime.strptime(current_date, "%Y-%m-%d")
        current_weekday_num = _weekday_to_num(current_weekday)
        
        # 基础相对时间
        if relative_time == '今天':
            return current_dt
        elif relative_time == '昨天':
            return current_dt - timedelta(days=1)
        elif relative_time == '明天':
            return current_dt + timedelta(days=1)
        elif relative_time == '后天':
            return current_dt + timedelta(days=2)
        elif relative_time == '大后天':
            return current_dt + timedelta(days=3)
        elif relative_time == '前天':
            return current_dt - timedelta(days=2)
        elif relative_time == '大前天':
            return current_dt - timedelta(days=3)
        
        # 相对周词
        elif '这周' in relative_time or '本周' in relative_time or relative_time.startswith('本'):
            # 提取星期几
            if '这周' in relative_time:
                weekday_char = relative_time.replace('这周', '')
            elif '本周' in relative_time:
                weekday_char = relative_time.replace('本周', '')
            else:
                weekday_char = relative_time.replace('本', '')
            
            target_weekday = _char_to_weekday_num(weekday_char)
            # 本周的星期X
            days_ahead = (target_weekday - current_weekday_num) % 7
            if days_ahead == 0:  # 如果今天就是目标星期，就是今天
                return current_dt
            else:
                return current_dt + timedelta(days=days_ahead)
        
        elif relative_time == '下周':
            # 单独的"下周"（没有具体星期几），默认下周一
            target_weekday = 0  # 周一
            days_ahead = (target_weekday - current_weekday_num) % 7
            if days_ahead == 0:  # 如果今天就是周一，下周一就是7天后
                return current_dt + timedelta(days=7)
            else:
                return current_dt + timedelta(days=days_ahead)
        
        elif '下周' in relative_time:
            # 提取星期几
            weekday_char = relative_time.replace('下周', '')
            target_weekday = _char_to_weekday_num(weekday_char)
            # 下一个星期X（不是下下周）
            days_ahead = (target_weekday - current_weekday_num) % 7
            if days_ahead == 0:  # 如果今天就是目标星期，下个星期X是7天后
                return current_dt + timedelta(days=7)
            else:
                return current_dt + timedelta(days=days_ahead)
        
        elif '下下' in relative_time:
            # 提取星期几
            weekday_char = relative_time.replace('下下', '')
            target_weekday = _char_to_weekday_num(weekday_char)
            # 下下周的星期X
            days_ahead = (target_weekday - current_weekday_num) % 7
            if days_ahead == 0:  # 如果今天就是目标星期，下下周的星期X是14天后
                return current_dt + timedelta(days=14)
            else:
                return current_dt + timedelta(days=7 + days_ahead)
        
        elif '上周' in relative_time:
            # 提取星期几
            weekday_char = relative_time.replace('上周', '')
            target_weekday = _char_to_weekday_num(weekday_char)
            # 上周的星期X
            days_ahead = (target_weekday - current_weekday_num) % 7
            if days_ahead == 0:  # 如果今天就是目标星期，上周的星期X是7天前
                return current_dt - timedelta(days=7)
            else:
                return current_dt - timedelta(days=7 - days_ahead)
        
        # 相对天数
        elif '天后' in relative_time or '天之后' in relative_time:
            match = re.search(r'(\d+)天', relative_time)
            if match:
                days = int(match.group(1))
                return current_dt + timedelta(days=days)
        
        elif '天前' in relative_time:
            match = re.search(r'(\d+)天', relative_time)
            if match:
                days = int(match.group(1))
                return current_dt - timedelta(days=days)
        
        elif '过' in relative_time and '天' in relative_time:
            match = re.search(r'过(\d+)天', relative_time)
            if match:
                days = int(match.group(1))
                return current_dt + timedelta(days=days)
        
        # 相对月份
        elif relative_time == '下个月':
            # 下个月1号
            if current_dt.month == 12:
                # 跨年
                return datetime(current_dt.year + 1, 1, 1)
            else:
                return datetime(current_dt.year, current_dt.month + 1, 1)
        
        # 其他相对时间（"最近"、"前几天"、"过几天"等）无法准确计算，返回None
        return None
        
    except Exception as e:
        logger.bind(tag=TAG).error(f"计算日期失败: {relative_time}, 错误: {e}")
        return None


def clean_relative_time_words_with_llm(content: str, llm_client) -> str:
    """
    用LLM清理内容中的相对时间词，保证句子通顺
    
    Args:
        content: 需要清理的内容
        llm_client: LLM客户端
    
    Returns:
        str: 清理后的内容
    """
    if not llm_client:
        logger.bind(tag=TAG).warning("LLM客户端未提供，使用简单清理")
        return _simple_clean_relative_time_words(content)
    
    try:
        prompt = f"""请从以下文本中移除所有相对时间词（如"今天"、"明天"、"下周二"、"3天后"、"下个月"等），但保持句子通顺自然。

要求：
1. 移除所有相对时间词
2. 保持句子意思不变
3. 保证句子通顺自然
4. 不要添加任何额外内容

示例：
输入：用户下周三有英语考试
输出：用户有英语考试

输入：用户3天后要出差
输出：用户要出差

输入：用户今天去爬山了
输出：用户去爬山了

输入：用户昨天跟同学去看电影了
输出：用户跟同学去看电影了

输入：用户下个月要去旅游
输出：用户要去旅游

现在请处理：
输入：{content}
输出："""
        
        response = llm_client.response_no_stream("", prompt, max_tokens=200)
        cleaned = response.strip()
        
        # 如果LLM返回为空或异常，使用简单清理
        if not cleaned or len(cleaned) < len(content) * 0.5:  # 如果清理后太短，可能有问题
            logger.bind(tag=TAG).warning(f"LLM清理结果异常，使用简单清理: {cleaned}")
            return _simple_clean_relative_time_words(content)
        
        return cleaned
        
    except Exception as e:
        logger.bind(tag=TAG).error(f"LLM清理相对时间词失败: {e}，使用简单清理")
        return _simple_clean_relative_time_words(content)


def _simple_clean_relative_time_words(content: str) -> str:
    """
    简单清理相对时间词（备用方案）
    
    Args:
        content: 需要清理的内容
    
    Returns:
        str: 清理后的内容
    """
    cleaned = content
    
    # 移除相对时间词
    for pattern in RELATIVE_TIME_PATTERNS:
        cleaned = re.sub(pattern, '', cleaned)
    
    # 清理多余的空格和标点
    cleaned = re.sub(r'\s+', ' ', cleaned)  # 多个空格变一个
    cleaned = re.sub(r'\s*，\s*', '，', cleaned)  # 清理逗号前后的空格
    cleaned = cleaned.strip()
    
    return cleaned


def validate_and_correct_date(
    memory_text: str, 
    current_date: str, 
    current_weekday: str,
    llm_client
) -> Tuple[str, bool]:
    """
    验证并纠正记忆文本中的日期，并用LLM清理相对时间词
    
    Args:
        memory_text: 记忆文本，格式如 "2026-01-12: 用户下周二有英语考试"
        current_date: 当前日期，格式 "YYYY-MM-DD"
        current_weekday: 当前星期几，如"星期一"
        llm_client: LLM客户端，用于清理相对时间词
    
    Returns:
        Tuple[str, bool]: (处理后的文本, 是否进行了处理)
    """
    # 【步骤1】检测是否包含相对时间词
    if not has_time_information(memory_text):
        return memory_text, False  # 没有相对时间词，不验证，直接返回
    
    # 【步骤2】提取日期和内容
    date_match = re.match(r'^(\d{4}-\d{2}-\d{2}):\s*(.+)', memory_text)
    if not date_match:
        return memory_text, False
    
    extracted_date_str = date_match.group(1)
    content = date_match.group(2)
    
    # 【步骤3】检测相对时间词并计算正确日期
    detected_relative = detect_relative_time_word(content)
    if not detected_relative:
        logger.bind(tag=TAG).debug(f"检测到需要验证但无法提取相对时间词: {memory_text}")
        return memory_text, False
    
    correct_date = calculate_correct_date(detected_relative, current_date, current_weekday)
    if not correct_date:
        logger.bind(tag=TAG).debug(f"无法计算相对时间词的日期: {detected_relative}, {memory_text}")
        return memory_text, False
    
    # 【步骤4】比较日期
    try:
        extracted_date = datetime.strptime(extracted_date_str, "%Y-%m-%d")
        date_diff = abs((extracted_date - correct_date).days)
        
        # 【步骤5】确定最终日期
        # 对于相对时间词的计算，应该是精确的，任何差异都应该纠正
        if date_diff > 0:
            # 日期错误，使用正确日期
            final_date_str = correct_date.strftime("%Y-%m-%d")
            was_corrected = True
            logger.bind(tag=TAG).warning(
                f"日期已纠正: {extracted_date_str} → {final_date_str} "
                f"(相对时间词: {detected_relative}, 差异: {date_diff}天)"
            )
        else:
            # 日期正确，使用原日期
            final_date_str = extracted_date_str
            was_corrected = False
            logger.bind(tag=TAG).debug(
                f"日期验证通过: {extracted_date_str} (相对时间词: {detected_relative}, 差异: {date_diff}天)"
            )
    except Exception as e:
        logger.bind(tag=TAG).error(f"日期比较失败: {e}, {memory_text}")
        return memory_text, False
    
    # 【步骤6】用LLM清理相对时间词
    cleaned_content = clean_relative_time_words_with_llm(content, llm_client)
    
    # 【步骤7】组合最终文本
    final_text = f"{final_date_str}: {cleaned_content}"
    
    return final_text, True  # 已处理（可能纠正了日期，并清理了相对时间词）

