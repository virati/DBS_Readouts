# %%
from dbread.utils.clips import clip
import matplotlib.pyplot as plt
import numpy as np
import dbread.sys.rosys as rosys
from dbread.utils.functions import innerprod
%load_ext autoreload
%autoreload 2

# %%

putative = rosys.rosys(num_nodes=10, num_probes=10, num_behaviors=3)
putative.set_H(innerprod, clip(np.eye(10), clip_num=2)).set_Γ(
    innerprod, clip(np.ones((10, 3)), clip_num=0))


putative.plot_graph()
putative.plot_coverage()
putative.plot_connectivity()
# %%
putative.plot_x()

# %%
H_vec_clipped = clip(np.eye(10), clip_num=2)
plt.imshow(H_vec_clipped)
plt.show()
putative.gen_synth_states(T=1000)
putative.set_H(innerprod, H_vec_clipped).measure(plot=True)
putative.behave(plot=True)

# %%
putative.train_readout().test_readout()
