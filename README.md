The project's codebase is quite large to bundle into a single file. Therefore, a folder has been created - project_code _ with my code. Every other file outside this folder is an example from the authors of RCareWorld, a config file or assets.

Only the folder **project_code** contains custom code

naive_rl is being finalised and readied for training. Due to issues with some of the api methods, custom placeholder implementations of some logic - grasp sponge for example - is provided. Once everything is sorted out, training will commence.

THat said, follow the installation guidelines at https://github.com/empriselab/RCareWorld/tree/phy-robo-care to setup the environment. THen, unzip this folder into the RCAREWORLD > folder. 

THe main files are:

1. reach_sponge_main.py: Contains code for the subtask: grasp_sponge. Ready for training.

2. dip_sponge.py: Contains code for dipping the sponge in the water tank and proceeding to the mannequin area.  Almost for training.

3. behaviour_cloning_reach_sponge.py: Contains behavioural cloning code to initialise reach_sponge. Ready for pre-training.

4. behaviour_cloning_rdip_water_tank: Still being written. Available for pre-training soon.

5. Perception.py: Perception system to locate objects and test actions before defining them in the action space

6. requirements.txt file

7. logs and models folder for logs and models (a2c, ppo, and later sac) respectively.

8. Images folder
