# 读取文件
with open('web_interface.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 替换第 1006-1021 行
new_section = '''        elif provider == "doubao":
            api_key = os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
            # Fallback to config if not in environment
            if not api_key and hasattr(analyzer, 'config'):
                api_key = analyzer.config.get("api_key", "")
        
        if api_key and api_key != "your-openai-api-key-here" and api_key != "":
            # Return masked version for security
            masked_key = (
                api_key[:3] + "..." + api_key[-3:] if len(api_key) > 12 else "***"
            )
            return jsonify({"has_key": True, "masked_key": masked_key})
        else:
            return jsonify({"has_key": False})
'''

# 写入修复后的文件
with open('web_interface.py', 'w', encoding='utf-8') as f:
    f.writelines(lines[:1006])
    f.write(new_section)
    f.writelines(lines[1022:])

print("修复完成")
