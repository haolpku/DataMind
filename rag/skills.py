"""
Skills 模块: 自定义工具/技能

每个 Skill 就是一个 Python 函数，包装成 FunctionTool 后注册到 Agent。
Agent 会根据用户问题自动决定是否调用。

如何添加新 Skill:
  1. 写一个普通 Python 函数
  2. 函数的 docstring 是 Agent 判断何时调用的依据 (务必写清楚)
  3. 参数需要有类型标注 (Agent 靠类型标注传参)
  4. 用 FunctionTool.from_defaults(fn=your_func) 包装
  5. 加入 get_all_skills() 的返回列表
"""

import datetime
import math
import subprocess
from llama_index.core.tools import FunctionTool


# ============================================================
# Skill 1: 获取当前时间
# ============================================================
def get_current_time() -> str:
    """获取当前日期和时间。当用户问"现在几点"、"今天几号"、"当前日期"等时间相关问题时使用。"""
    now = datetime.datetime.now()
    return now.strftime("当前时间: %Y年%m月%d日 %H:%M:%S (星期%w)")


# ============================================================
# Skill 2: 数学计算器
# ============================================================
def calculator(expression: str) -> str:
    """安全的数学计算器。当用户需要精确的数学计算时使用，如加减乘除、幂运算、三角函数等。
    expression: 数学表达式，如 '2+3*4', 'sqrt(144)', 'sin(3.14/2)'

    支持的函数: sqrt, sin, cos, tan, log, log10, ceil, floor, abs, pow, pi, e
    """
    safe_dict = {
        "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
        "tan": math.tan, "log": math.log, "log10": math.log10,
        "ceil": math.ceil, "floor": math.floor, "abs": abs,
        "pow": pow, "pi": math.pi, "e": math.e,
        "round": round, "max": max, "min": min,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        return f"计算: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"


# ============================================================
# Skill 3: 文本分析
# ============================================================
def analyze_text(text: str) -> str:
    """分析一段文本的基本统计信息。当用户需要统计字数、段落数等文本特征时使用。
    text: 要分析的文本内容
    """
    char_count = len(text)
    char_no_space = len(text.replace(" ", "").replace("\n", ""))
    lines = text.strip().split("\n")
    line_count = len(lines)
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    para_count = len(paragraphs)
    words = text.split()
    word_count = len(words)

    return (
        f"文本分析结果:\n"
        f"- 总字符数: {char_count}\n"
        f"- 字符数(不含空格): {char_no_space}\n"
        f"- 单词/词语数: {word_count}\n"
        f"- 行数: {line_count}\n"
        f"- 段落数: {para_count}"
    )


# ============================================================
# Skill 4: 单位换算
# ============================================================
def unit_convert(value: float, from_unit: str, to_unit: str) -> str:
    """单位换算工具。支持长度、重量、温度的常见换算。
    value: 数值
    from_unit: 源单位 (如 km, m, cm, mm, mile, ft, inch, kg, g, lb, oz, celsius, fahrenheit, kelvin)
    to_unit: 目标单位
    """
    conversions = {
        ("km", "m"): lambda v: v * 1000,
        ("m", "km"): lambda v: v / 1000,
        ("m", "cm"): lambda v: v * 100,
        ("cm", "m"): lambda v: v / 100,
        ("m", "mm"): lambda v: v * 1000,
        ("mm", "m"): lambda v: v / 1000,
        ("km", "mile"): lambda v: v * 0.621371,
        ("mile", "km"): lambda v: v * 1.60934,
        ("m", "ft"): lambda v: v * 3.28084,
        ("ft", "m"): lambda v: v / 3.28084,
        ("inch", "cm"): lambda v: v * 2.54,
        ("cm", "inch"): lambda v: v / 2.54,
        ("kg", "g"): lambda v: v * 1000,
        ("g", "kg"): lambda v: v / 1000,
        ("kg", "lb"): lambda v: v * 2.20462,
        ("lb", "kg"): lambda v: v / 2.20462,
        ("oz", "g"): lambda v: v * 28.3495,
        ("g", "oz"): lambda v: v / 28.3495,
        ("celsius", "fahrenheit"): lambda v: v * 9 / 5 + 32,
        ("fahrenheit", "celsius"): lambda v: (v - 32) * 5 / 9,
        ("celsius", "kelvin"): lambda v: v + 273.15,
        ("kelvin", "celsius"): lambda v: v - 273.15,
    }

    f, t = from_unit.lower(), to_unit.lower()
    if (f, t) in conversions:
        result = conversions[(f, t)](value)
        return f"{value} {from_unit} = {result:.4g} {to_unit}"
    return f"不支持的换算: {from_unit} -> {to_unit}"


# ============================================================
# 注册所有 Skills
# ============================================================
def get_all_skills() -> list:
    """返回所有可用的 Skill 工具列表"""
    return [
        FunctionTool.from_defaults(fn=get_current_time),
        FunctionTool.from_defaults(fn=calculator),
        FunctionTool.from_defaults(fn=analyze_text),
        FunctionTool.from_defaults(fn=unit_convert),
    ]
