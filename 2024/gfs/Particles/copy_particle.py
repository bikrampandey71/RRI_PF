import os
import shutil

if __name__ == "__main__":

    end = int(input("Enter the particle number (default = 64): ") or 64)

    Base_Particle = os.path.join(".", "Particle00001")
    if not os.path.exists(Base_Particle):
        print(f"Error: The folder '{Base_Particle}' does not exist to copy from.")

    for i in range(2, end+1):
        folder_name = 'Particle' + f"{i:05}"
        folder_path = os.path.join(".", folder_name)
        try:
            shutil.copytree(Base_Particle, folder_path)
            print(f"Copied: {folder_path}")
        except Exception as e:
            print(f"Failed to create {folder_path}: {e}")


