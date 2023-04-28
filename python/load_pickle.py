#!/usr/bin/env python3
import pandas as pd

datafiles = [
"data/blume_capel_pickles/BlumeCapelSq2DEigvals_L=100_D=0_T=0.84689_mc_steps=300_n_samples=100_n_runs=1000.pickle",
"data/blume_capel_pickles/BlumeCapelSq2DEigvals_L=100_D=0_T=1.016268_mc_steps=300_n_samples=100_n_runs=1000.pickle",
"data/blume_capel_pickles/BlumeCapelSq2DEigvals_L=100_D=0_T=1.185646_mc_steps=300_n_samples=100_n_runs=1000.pickle",
"data/blume_capel_pickles/BlumeCapelSq2DEigvals_L=100_D=0_T=1.3550240000000002_mc_steps=300_n_samples=100_n_runs=1000.pickle",
"data/blume_capel_pickles/BlumeCapelSq2DEigvals_L=100_D=0_T=1.524402_mc_steps=300_n_samples=100_n_runs=1000.pickle",
"data/blume_capel_pickles/BlumeCapelSq2DEigvals_L=100_D=0_T=1.609091_mc_steps=300_n_samples=100_n_runs=1000.pickle",
"data/blume_capel_pickles/BlumeCapelSq2DEigvals_L=100_D=0_T=1.6514355_mc_steps=300_n_samples=100_n_runs=1000.pickle",
"data/blume_capel_pickles/BlumeCapelSq2DEigvals_L=100_D=0_T=1.69378_mc_steps=300_n_samples=100_n_runs=1000.pickle",
"data/blume_capel_pickles/BlumeCapelSq2DEigvals_L=100_D=0_T=1.7361244999999998_mc_steps=300_n_samples=100_n_runs=1000.pickle",
"data/blume_capel_pickles/BlumeCapelSq2DEigvals_L=100_D=0_T=1.778469_mc_steps=300_n_samples=100_n_runs=1000.pickle",
"data/blume_capel_pickles/BlumeCapelSq2DEigvals_L=100_D=0_T=1.8631580000000003_mc_steps=300_n_samples=100_n_runs=1000.pickle",
"data/blume_capel_pickles/BlumeCapelSq2DEigvals_L=100_D=0_T=1.9478469999999999_mc_steps=300_n_samples=100_n_runs=1000.pickle",
"data/blume_capel_pickles/BlumeCapelSq2DEigvals_L=100_D=0_T=11.00957_mc_steps=300_n_samples=100_n_runs=1000.pickle",
"data/blume_capel_pickles/BlumeCapelSq2DEigvals_L=100_D=0_T=2.032536_mc_steps=300_n_samples=100_n_runs=1000.pickle",
"data/blume_capel_pickles/BlumeCapelSq2DEigvals_L=100_D=0_T=2.117225_mc_steps=300_n_samples=100_n_runs=1000.pickle",
"data/blume_capel_pickles/BlumeCapelSq2DEigvals_L=100_D=0_T=2.2019140000000004_mc_steps=300_n_samples=100_n_runs=1000.pickle",
"data/blume_capel_pickles/BlumeCapelSq2DEigvals_L=100_D=0_T=2.2866030000000004_mc_steps=300_n_samples=100_n_runs=1000.pickle",
"data/blume_capel_pickles/BlumeCapelSq2DEigvals_L=100_D=0_T=2.371292_mc_steps=300_n_samples=100_n_runs=1000.pickle",
"data/blume_capel_pickles/BlumeCapelSq2DEigvals_L=100_D=0_T=2.455981_mc_steps=300_n_samples=100_n_runs=1000.pickle",
"data/blume_capel_pickles/BlumeCapelSq2DEigvals_L=100_D=0_T=2.54067_mc_steps=300_n_samples=100_n_runs=1000.pickle",
"data/blume_capel_pickles/BlumeCapelSq2DEigvals_L=100_D=0_T=4.23445_mc_steps=300_n_samples=100_n_runs=1000.pickle",
"data/blume_capel_pickles/BlumeCapelSq2DEigvals_L=100_D=0_T=5.92823_mc_steps=300_n_samples=100_n_runs=1000.pickle",
"data/blume_capel_pickles/BlumeCapelSq2DEigvals_L=100_D=0_T=7.62201_mc_steps=300_n_samples=100_n_runs=1000.pickle",
"data/blume_capel_pickles/BlumeCapelSq2DEigvals_L=100_D=0_T=9.31579_mc_steps=300_n_samples=100_n_runs=1000.pickle"
]

for datafile in datafiles:
    print(datafile)

    df: list = pd.read_pickle(datafile)



    break
