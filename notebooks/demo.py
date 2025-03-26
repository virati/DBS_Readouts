# %%
from dbread.utils.functions import innerprod
import dbread.sys.rosys as rosys
import numpy as np
import matplotlib.pyplot as plt
from dbread.utils.clips import clip
from dbread.viz.assessment import plot_against_Γ

%load_ext autoreload
%autoreload 2

# %%

# %%
num_nodes = 20
num_probes = 20
num_behaviors = 1

# %%
putative = rosys.rosys(num_nodes=num_nodes,
                       num_probes=num_probes, num_behaviors=num_probes)
putative.set_H(innerprod, clip(np.eye(num_nodes), clip_num=2)).set_Γ(
    innerprod, clip(np.random.normal(0, 1, size=(num_nodes, 1)), clip_num=0))


putative.plot_graph()
putative.plot_coverage()
putative.plot_connectivity()
# %%
putative.plot_x()

# %%
H_vec_clipped = clip(np.eye(num_probes), clip_num=0)
plt.imshow(H_vec_clipped)
plt.show()
putative.gen_synth_states(T=1000)
putative.set_H(innerprod, H_vec_clipped).measure(plot=True)
putative.behave(plot=True)

# %%
assessment = putative.train_readout().test_readout()
print(assessment)
# %%

# %% Loop through clips
results = []
alignment = []
H_coeffs = np.eye(num_probes)
for clip_num in range(0, num_probes):
    putative.set_H(innerprod, clip(
        H_coeffs, clip_num=clip_num)).measure(plot=False)
    putative.behave(plot=False)
    assessment = putative.train_readout().test_readout()
    results.append(assessment[0].statistic)
    alignment.append(assessment[1])

# %%
plot_against_Γ(putative, results, label="Accuracy")
plot_against_Γ(putative, alignment, label="Alignment")
