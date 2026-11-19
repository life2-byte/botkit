"""A tiny real MCP server (stdio) used by the tests."""
try:
    from mcp.server.mcpserver import MCPServer as _Server  # mcp 2.x
except ImportError:
    from mcp.server.fastmcp import FastMCP as _Server  # mcp 1.x

server = _Server("demo")


@server.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


@server.tool()
def boom() -> str:
    """Always fails."""
    raise RuntimeError("kaboom")


if __name__ == "__main__":
    server.run()
