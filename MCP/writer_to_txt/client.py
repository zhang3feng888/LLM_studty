import os
import asyncio
from typing import Optional
from contextlib import AsyncExitStack
import json
import traceback
 
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
 
from openai import OpenAI
from dotenv import load_dotenv
 
load_dotenv()  # 加载环境变量从 .env
 
class MCPClient:
    def __init__(self):
        # 初始化会话和客户端对象
        self.session: Optional[ClientSession] = None # 会话对象
        self.exit_stack = AsyncExitStack() # 退出堆栈
        self.openai = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1") 
        self.model="Qwen/Qwen3-4B"
 
    def get_response(self, messages: list,tools: list):
        response = self.openai.chat.completions.create(
            model=self.model,
            max_tokens=1000,
            messages=messages,
            tools=tools,
        )
        return response
    
    async def get_tools(self):
        # 列出可用工具
        response = await self.session.list_tools()
        available_tools = [{ 
            "type":"function",
            "function":{
                "name": tool.name,
                "description": tool.description, # 工具描述
                "parameters": tool.inputSchema  # 工具输入模式
            }
        } for tool in response.tools]
        
        return available_tools
        
    
    async def connect_to_server(self, server_script_path: str):
        """连接到 MCP 服务器
    
        参数:
            server_script_path: 服务器脚本路径 (.py 或 .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("服务器脚本必须是 .py 或 .js 文件")
            
        command = "python" if is_python else "node"
        # 创建 StdioServerParameters 对象
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )
        
        # 使用 stdio_client 创建与服务器的 stdio 传输
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        
        # 解包 stdio_transport，获取读取和写入句柄
        self.stdio, self.write = stdio_transport
        
        # 创建 ClientSession 对象，用于与服务器通信
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        
        # 初始化会话
        await self.session.initialize()
        # 列出可用工具
        response = await self.session.list_tools()
        tools = response.tools
        print("\n连接到服务器，工具列表:", [tool.name for tool in tools])
 
    async def process_query(self, query: str) -> str:
        """使用 OpenAI 和可用工具处理查询"""
        
        # 创建消息列表
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]
        
        # 列出可用工具
        available_tools = await self.get_tools()
        print(f"\n可用工具: {json.dumps([t['function']['name'] for t in available_tools], ensure_ascii=False)}")
        
        # 处理消息
        response = self.get_response(messages, available_tools)
 
        # 处理LLM响应和工具调用
        tool_results = []
        final_text = []
        for choice in response.choices:
            message = choice.message
            is_function_call = message.tool_calls
            
            # 如果不调用工具，则添加到 final_text 中
            if not is_function_call:
                final_text.append(message.content)
            # 如果是工具调用，则获取工具名称和输入
            else:
                #解包tool_calls
                tool_name = message.tool_calls[0].function.name
                tool_args = json.loads(message.tool_calls[0].function.arguments)
                print(f"准备调用工具: {tool_name}")
                print(f"参数: {json.dumps(tool_args, ensure_ascii=False, indent=2)}")
                
                try:
                    # 执行工具调用，获取结果
                    result = await self.session.call_tool(tool_name, tool_args)
                    print(f"\n工具调用返回结果类型: {type(result)}")
                    print(f"工具调用返回结果属性: {dir(result)}")
                    print(f"工具调用content类型: {type(result.content) if hasattr(result, 'content') else '无content属性'}")
                    
                    # 安全处理content
                    content = None
                    if hasattr(result, 'content'):
                        if isinstance(result.content, list):
                            content = "\n".join(str(item) for item in result.content)
                            print(f"将列表转换为字符串: {content}")
                        else:
                            content = str(result.content)
                            print(f"工具调用content值: {content}")
                    else:
                        content = str(result)
                        print(f"使用完整result作为字符串: {content}")
                    
                    tool_results.append({"call": tool_name, "result": content})
                    final_text.append(f"[调用工具 {tool_name} 参数: {json.dumps(tool_args, ensure_ascii=False)}]")
 
                    # 继续与工具结果进行对话
                    if message.content and hasattr(message.content, 'text'):
                        messages.append({
                          "role": "assistant",
                          "content": message.content
                        })
                    
                    # 将工具调用结果添加到消息
                    messages.append({
                        "role": "user", 
                        "content": content
                    })
                    
                    # 获取下一个LLM响应
                    print("获取下一个LLM响应...")
                    response = self.get_response(messages, available_tools)
                    # 将结果添加到 final_text
                    content = response.choices[0].message.content or ""
                    final_text.append(content)
                except Exception as e:
                    print(f"\n工具调用异常: {str(e)}")
                    print(f"异常详情: {traceback.format_exc()}")
                    final_text.append(f"工具调用失败: {str(e)}")
 
        return "\n".join(final_text)
 
    async def chat_loop(self):
        """运行交互式聊天循环（没有记忆）"""
        print("\nMCP Client 启动!")
        print("输入您的查询或 'quit' 退出.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                
                if query.lower() == 'quit':
                    break
                    
                print("\n处理查询中...")
                response = await self.process_query(query)
                print("\n" + response)
                    
            except Exception as e:
                print(f"\n错误: {str(e)}")
                print(f"错误详情: {traceback.format_exc()}")
    
    async def cleanup(self):
        """清理资源"""
        await self.exit_stack.aclose() 
 
async def main():
    """
    主函数：初始化并运行 MCP 客户端
    此函数执行以下步骤：
    1. 检查命令行参数是否包含服务器脚本路径
    2. 创建 MCPClient 实例
    3. 连接到指定的服务器
    4. 运行交互式聊天循环
    5. 在结束时清理资源
    用法：
    python client.py <path_to_server_script>
    """
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)
    
    # 创建 MCPClient 实例
    client = MCPClient()
    try:
        # 连接到服务器
        await client.connect_to_server(sys.argv[1])
        # 运行聊天循环
        await client.chat_loop()
    finally:
        # 确保在任何情况下都清理资源
        await client.cleanup()
 
if __name__ == "__main__":
    import sys
    asyncio.run(main())