from wxauto import WeChat
from openai import OpenAI
import time

# 初始化 OpenAI 客户端
client = OpenAI(
    api_key="sk-bkg8j74es2n07gejunjtqnulnj2u3ce699fgp07cdoi5ar4s",
    base_url="https://api.aihao123.cn/luomacode-api/open-api/v1"
)

# 初始化 WeChat 客户端
wx = WeChat()

# 监听列表
listen_list = ['母亲']

def generate_response(prompt):
    """调用 OpenAI 模型生成回复"""
    response = client.chat.completions.create(
        messages=[
            {'role': 'user', 'content': prompt},
        ],
        model='gpt-3.5-turbo',
        stream=False
    )
    return response.choices[0].message.content

def get_latest_message(name):
    """获取指定联系人的最新消息"""
    wx.ChatWith(name)
    msgs = wx.GetAllMessage()  # 调用方法并获取消息列表
    if msgs:
        return msgs[-1]  # 返回最新的一条消息
    return None

def check_messages():
    """检查监听列表中的联系人是否有新消息，并自动回复"""
    for name in listen_list:
        latest_msg = get_latest_message(name)
        if latest_msg:
            # 打印调试信息，检查 latest_msg 的格式
            print(f"Latest message: {latest_msg}")
            # 假设 latest_msg 是一个 FriendMessage 对象，提取消息内容
            if hasattr(latest_msg, 'content'):
                content = latest_msg.content
                sender = name  # 直接使用监听列表中的名称作为发送者
                print(f"Received message from {sender}: {content}")
                # 生成回复
                response = generate_response(content)
                print(f"Sending response to {sender}: {response}")
                # 发送回复
                wx.SendMsg(response, name)
            else:
                print(f"Unexpected message format: {latest_msg}")

if __name__ == '__main__':
    while True:
        check_messages()
        time.sleep(5)  # 每 5 秒检查一次消息
