# %%
from dbread.utils.functions import innerprod
import dbread.sys.rosys as rosys
import numpy as np
import matplotlib.pyplot as plt
from dbread.utils.clips import clip
%load_ext autoreload
%autoreload 2

# %%

# %%

putative = rosys.rosys(num_nodes=10, num_probes=10, num_behaviors=1)
putative.set_H(innerprod, clip(np.eye(10), clip_num=2)).set_Γ(
    innerprod, clip(np.random.normal(0, 1, size=(10, 1)), clip_num=0))


putative.plot_graph()
putative.plot_coverage()
putative.plot_connectivity()
# %%
putative.plot_x()

# %%
H_vec_clipped = clip(np.eye(10), clip_num=0)
plt.imshow(H_vec_clipped)
plt.show()
putative.gen_synth_states(T=1000)
putative.set_H(innerprod, H_vec_clipped).measure(plot=True)
putative.behave(plot=True)

# %%
assessment = putative.train_readout().test_readout()
