import tkinter as tk
from tkinter import ttk

# 关系计算函数
def calculate_relation():
    gender = gender_var.get()
    call_style = call_style_var.get()
    region = region_var.get()
    relation_input = relation_entry.get()

    # 简单示例：解析输入并返回结果
    if relation_input == "妈妈的女儿的儿子":
        if call_style == "我称呼对方":
            result = "外孙" if gender == "男" else "外孙女"
        else:
            result = "孙子" if gender == "男" else "孙女"
    else:
        result = "未知关系"

    result_label.config(text=f"计算结果：{result}")

# 创建主窗口
root = tk.Tk()
root.title("亲戚关系计算器")

# 性别选择
gender_var = tk.StringVar(value="男")
ttk.Label(root, text="我的性别：").grid(column=0, row=0, sticky=tk.W)
ttk.Radiobutton(root, text="男", variable=gender_var, value="男").grid(column=1, row=0)
ttk.Radiobutton(root, text="女", variable=gender_var, value="女").grid(column=2, row=0)

# 称呼方式选择
call_style_var = tk.StringVar(value="我称呼对方")
ttk.Label(root, text="称呼方式：").grid(column=0, row=1, sticky=tk.W)
ttk.Radiobutton(root, text="我称呼对方", variable=call_style_var, value="我称呼对方").grid(column=1, row=1)
ttk.Radiobutton(root, text="对方称呼我", variable=call_style_var, value="对方称呼我").grid(column=2, row=1)

# 区域选择
region_var = tk.StringVar(value="通用")
ttk.Label(root, text="区域选择：").grid(column=0, row=2, sticky=tk.W)
ttk.Radiobutton(root, text="通用", variable=region_var, value="通用").grid(column=1, row=2)
ttk.Radiobutton(root, text="北方", variable=region_var, value="北方").grid(column=2, row=2)
ttk.Radiobutton(root, text="南方", variable=region_var, value="南方").grid(column=3, row=2)
ttk.Radiobutton(root, text="文言", variable=region_var, value="文言").grid(column=4, row=2)

# 输入框
ttk.Label(root, text="关系输入：").grid(column=0, row=3, sticky=tk.W)
relation_entry = ttk.Entry(root, width=30)
relation_entry.grid(column=1, row=3, columnspan=4)

# 计算按钮
calculate_button = ttk.Button(root, text="计算关系", command=calculate_relation)
calculate_button.grid(column=1, row=4, columnspan=2)

# 清空按钮
clear_button = ttk.Button(root, text="清空", command=lambda: relation_entry.delete(0, tk.END))
clear_button.grid(column=3, row=4, columnspan=2)

# 结果显示
result_label = ttk.Label(root, text="计算结果：")
result_label.grid(column=0, row=5, columnspan=5)

# 启动主循环
root.mainloop()
