import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack
from typing import Optional
 
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters, stdio_client
from openai import OpenAI
 
sys.stdout.reconfigure(encoding='utf-8')
os.environ['PYTHONIOENCODING'] = 'utf-8'
 
# 加载 .env 文件中的变量    OPENAI_API_KEY、BASE_URL、MODEL
# sys.stdout.reconfigure(encoding='utf-8') 保障中文正常显示
load_dotenv()
 
class MCPClient:
    def __init__(self):
        """初始化MCP客户端"""
        self.exit_stack = AsyncExitStack() # 异步上下文资源管理器
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.base_url = os.getenv('BASE_URL')
        self.model = os.getenv('MODEL')
        if not self.openai_api_key:
            raise ValueError("❌未找到OpenAI API Key，请在.env文件中设置OPENAI_API_KEY")
 
        self.client = OpenAI(api_key=self.openai_api_key, base_url=self.base_url)
        # self.session 实例变量，用于保存连接到 MCP 服务端后的通信会话
        self.session: Optional[ClientSession] = None

    async def connect_to_server(self, server_script_path):
        # server_script_path：server.py的路径
        """连接到MCP服务器"""
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not is_python and not is_js:
            raise ValueError("❌不支持的脚本类型，请使用Python或JavaScript脚本")
 
        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None,
        )
 
        # 启动 MCP 服务器并建立通信
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
 
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
 
        await self.session.initialize()
 
        #  列出 MCP服务器上的工具
        response = await self.session.list_tools()
        tools = response.tools
        print("\n已连接到服务器，支持以下工具:", [tool.name for tool in tools])
 
 
    async def process_query(self, query):
        """
        使用大模型处理查询并调用可用的 MCP 工具 (Function Calling)
        """
        messages = [{"role": "system", "content": "你是一个智能助手，帮助用户回答问题。"},
        {"role": "user", "content": query}]
        response = await self.session.list_tools()
 
        available_tools = [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema,
            }
        } for tool in response.tools]
 
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=available_tools
        )
 
        content = response.choices[0]
        if content.finish_reason == "tool_calls":
            tool_call = content.message.tool_calls[0]
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
 
            # 执行工具
            result = await self.session.call_tool(function_name, function_args)
            print(f"\n\n[Calling tool {function_name} with args {function_args}]\n\n")
            # 将模型返回的调用哪个工具数据和工具执行完成后的数据都存入messages中
            result_content = result.content[0].text
            messages.append(content.message.model_dump())
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": result_content,
            })
 
            # 将上面的结果再返回给大模型用于生产最终的结果
            return result_content
 
        return content.message.content.strip()
 
 
 
    async def chat_loop(self):
        """运行交互式聊天循环"""
        print("✅MCP 客户端已启动！输入 'quit' 退出")
 
        while True:
            try:
                query = input("输入你的问题：").strip()
                if query.lower() == 'quit':
                    break
 
                response = await self.process_query(query)
 
                print(f"openai：{response}")
            except Exception as e:
                print(f"发生错误：{e}")
 
    async def cleanup(self):
        """清理资源"""
        await self.exit_stack.aclose()
 
 
async def main():
    # argv 两个参数。 argv[0]：client.py   argv[1]：server.py
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)
    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()
 
 
if __name__ == "__main__":
    asyncio.run(main())