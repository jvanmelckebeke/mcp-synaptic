"""Main entry point for MCP Synaptic server."""

import asyncio
import sys

from mcp_synaptic import SynapticServer, Settings


async def main() -> None:
    """Main entry point."""
    try:
        # Load settings
        settings = Settings()
        
        # Create and start server
        server = SynapticServer(settings)
        await server.start()
        
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
