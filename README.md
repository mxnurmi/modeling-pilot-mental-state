# modeling-pilot-mental-state

This project was made in collaboration with Saab and Aalto University. The goal of the project was to build a Reinforcement Learning tool which allows us to model the stress state of an airline pilot.

Modeling tools
==============

The modeling code can be ran from the ./runner.py file, with the run() function. The run function has three alternative ways it can be used to produce results of the RL agent's behaviour

In the first case we can use the agent to produce a real-time visualization of it running in the given task environment:

`run(stress="attribute", scenario_number="five")`

Here you can choose what type of stress-model we are using and which one of the predetermined scenarios we can run. All of the scenarios can be initiated through the config file which also provides parameters of each scenario. Upon running the above code, the Python will open a separate window which shows the agent running in the given scenario with stress modeled in real-time. 

In the second case we can do the same, but instead of opening a seprate window, we can save the gif animation into ./animations folder:

`run(save_animation=True, stress="attribute", scenario_number="five")`

And in the third scenario we can generate a datafile that contains the results of agent running predetermined amount of times with the various stress states saved. Each run starts with the agent's state reset. This is useful if we want to evaluate how stress behaves over multiple runs:

`run(simulate_agent=True, loops=40, stress="attribute", scenario_number="five")`

Note that even 100 loops can take a considerable amount of time on a basic computer as the agent will run about 40000 simulations at each step, which each loop will have multiple of.

Modeling in simulation in real-time
===================================

Finally, besides offering these modeling tools, the tool also offers a way the tool in XPlane (11), providing real-time stress evaluation for an user flying in the simulation.

This can be achieved by starting a flight in XPlane and then running the xplane_master.py
