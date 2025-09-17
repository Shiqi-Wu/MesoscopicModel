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


# Group Meeting

## 2025.09.10
1. accelerate the continuous time simulation for Glauber dynamics. ☑️
2. check with more trials for each param setting, also larger L. ☑️
3. since the convergence rate depends on h, T, the convergence time varies. thus, we need more simulation time. ☑️
4. add second order statistics evolution curve. ☑️
5. test with simple local Allen-Cahn Equation, check with different size of $\gamma$. ☑️
6. we may need assumptions for the MLP force network to be unique.
7. start with the ground truth initialization of J, to see the minimum loss. ☑️
8. not sure about the time thing in the data and nn training, to check.
9. landau energy as Lyapunov function? at least it's a local func when temperature is high.

## 2025.09.17
1. check why local pde solution is not correct. spatial resolution? ☑️
2. regularity retrictions for the function.
3. with larger L, check the true solution.
4. train and compare with different baselines.