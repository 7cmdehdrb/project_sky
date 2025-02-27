import asyncio
import websockets


async def send_message():
    uri = "ws://localhost:8765"
    async with websockets.connect(uri) as websocket:
        message = "Hello, WebSocket!"
        await websocket.send(message)
        print(f"Sent: {message}")


if __name__ == "__main__":
    asyncio.run(send_message())
