from stable_baselines3 import DDPG
from env import Environment
from config_reader import Config
from preprocess import DataModule
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

config = Config()
train_file_list = [f"dataset/train-{i}" for i in range(1, 51)]

dataModules = []
entries = []
for j in range(4):
    for file in train_file_list:
        config.train["train_bin_path"] = file
        dataModules.append(DataModule(config.train, config.preprocess))
        dataModules[-1].setup()
        entries.append(len(dataModules[-1].train))


class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self):
        self.logger.record("train/reward", self.locals["rewards"])
        return True


checkpoint_callback = CheckpointCallback(
    save_freq=1,
    save_path="./logs/",
    name_prefix="ddpg_nas",
    save_replay_buffer=True,
    save_vecnormalize=True,
)

env = Environment(dataModules, entries, config)
env = DummyVecEnv([lambda: env])
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
model = DDPG("MlpPolicy", env, verbose=1, tensorboard_log="logs/ddpg")
model.learn(
    total_timesteps=20000, callback=[TensorboardCallback(), checkpoint_callback]
)
model.save("ddpg_transformer")
