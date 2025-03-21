import asyncio
import pyvts

async def main():
    vts = pyvts.vts()
    
    # Keep trying to connect until successful
    while True:
        try:
            await vts.connect()
            print("Successfully connected to VTube Studio!")
            break
        except Exception as e:
            print(f"Failed to connect: {e}. Retrying in 5 seconds...")
            await asyncio.sleep(5)  # Wait for 5 seconds before retrying

    # Example: Move the character using available methods
    try:
        await vts.move_x(100)
        await vts.move_y(200)
        await vts.move_z(300)
        await vts.rotate(45)
        await vts.scale(1.0)
        print("Model moved successfully!")
    except Exception as e:
        print(f"Error moving the model: {e}")

    await vts.close()

if __name__ == "__main__":
    asyncio.run(main())
