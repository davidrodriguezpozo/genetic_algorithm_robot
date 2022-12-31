## Genetic algorithms with PyBullet

The goal of this project is to create a creature (`.rdf` file), which evolves with a goal in mind: travel the longest distance in the least time. With this objective, the robot can be self-trained and self-evolves to travel more distance with each genetic iteration. 


To see the results, run `pybullet` using

```python
p.connect(p.GUI)
# Rest of the code here...
```

See how with each training session (each new robot), the robots end up having different features, due to the random genetic mutations introduced in each step. 
