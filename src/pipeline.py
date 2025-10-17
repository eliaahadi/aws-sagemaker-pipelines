import subprocess

def run(cmd):
    print(f"\n$ {cmd}")
    subprocess.run(cmd, shell=True, check=True)

if __name__ == "__main__":
    run("python src/train.py")
    run("python src/register.py")
    run("python src/approve.py --version latest")
    print("\nPipeline complete. Run `make deploy` to start the real-time endpoint.")