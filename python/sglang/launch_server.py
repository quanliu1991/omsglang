import argparse
import multiprocessing as mp


from sglang.srt.server import ServerArgs, launch_server

if __name__ == "__main__":
    mp.set_start_method("spawn")
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)

    launch_server(server_args, None)
