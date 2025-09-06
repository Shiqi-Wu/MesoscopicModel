#  Structure

1. src/
    1. config.py
    2. glauber.py
    3. pde_solver.py
    4. networks.py
    5. utils.py
    6. kernels.py
    7. plot.py
    8. train.py
2. scripts/ 
    1. glauber_simulation/
        1. gaussian random field initialization
        2. align with the paper setting
        3. different_b (var)
        4. different_t (critical temperature)
        5. different_h
        6. different kernel range domain $\gamma$
        7. check with macro properties (check with pde)
    2. pde_compare/
        1. simple local allen-cahn
        2. nonlocal allen-cahn
        3. compare glauber, simple pde and nonlocal pde (for different settings)
    3. nn_train/
        1. train with simple cnn kernel and F=-Am+tanh(Bi+m+h)
        2. train with simple cnn kernel and mlp F
        3. onsager-like network structure for F
        4. visualize kernel and F
3. data/
    1. .gitkeep
4. results/
    1. .gitkeep
5. notes/
    1. tex files


# My Questions
1. why the shape is 502 from 0 to 5, why not 501?
2. plot for given spins
   1. python tests/test_animation.py 