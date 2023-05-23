import hydra
from isaacgymenvs.train import launch_rlg_hydra
from isaacgymenvs.tasks import isaacgym_task_map

from env.quad_route import QuadRoute

def register_task_map():
    isaacgym_task_map["QuadRoute"] = QuadRoute

@hydra.main(config_name="config", config_path="./cfg")
def main(cfg):
    launch_rlg_hydra(cfg)


if __name__ == "__main__":
    register_task_map()
    main()