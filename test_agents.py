import json
from agents_comparison import run_tool, extract_json

# 1. Тест калькулятора (tool_calc) и обработки синтаксических ошибок
assert run_tool("calc", "2*(3+4)") == "14", "Ошибка в базовом расчете"
assert "Ошибка" in run_tool("calc", "2+"), "Некорректное выражение должно возвращать строку с ошибкой"

# 2. Тест защиты от неизвестного инструмента
assert "неизвестный инструмент" in run_tool("weather", "Moscow"), "Неизвестный инструмент должен возвращать ошибку"

# 3. Тест парсера JSON (extract_json) с оберткой markdown и текстом
wrapped_response = 'Вот ответ модели:\n```json\n{"action": "search", "input": "refund", "answer": ""}\n```'
parsed = extract_json(wrapped_response)
assert parsed["action"] == "search" and parsed["input"] == "refund", "Парсер не смог извлечь JSON из обертки"

# 4. Тест цикла агента с замоканным LLM (mock_llm) и проверкой лимита шагов
def mock_llm(prompt: str) -> str:
    if "Действие:" not in prompt:
        return '{"action": "calc", "input": "100 + 250", "answer": ""}'
    return '{"action": "final", "input": "", "answer": "Итоговая сумма: 350"}'

def run_mock_agent(question: str, max_steps: int = 3) -> str:
    history = f"Вопрос: {question}\n"
    for _ in range(max_steps):
        step = extract_json(mock_llm(history))
        if step["action"] == "final":
            return step["answer"]
        res = run_tool(step["action"], step["input"])
        history += f"Действие: {step['action']}({step['input']}) -> {res}\n"
    return "Лимит шагов исчерпан"

assert run_mock_agent("Посчитай 100 + 250") == "Итоговая сумма: 350", "Сбой в цикле замоканного агента"

print("✅ Все юнит-тесты инструментов, парсера JSON и цикла агента успешно пройдены!")
