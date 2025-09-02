import seisbench.data as sbd
import seisbench.generate as sbg

data = sbd.ETHZ(sampling_rate=100)

generator = sbg.GenericGenerator(data)

print(generator)

import matplotlib.pyplot as plt

print("Number of examples:", len(generator))
sample = generator[200]
print("Example:", sample)

plt.plot(sample["X"].T)
plt.show()

generator.augmentation(sbg.RandomWindow(windowlen=3000))
generator.augmentation(sbg.Normalize(detrend_axis=-1, amp_norm_axis=-1))

print(generator)

sample = generator[200]
print("Example:", sample)

plt.plot(sample["X"].T)
plt.show()

generator.augmentation(
    sbg.ProbabilisticLabeller(
        label_columns=["trace_P1_arrival_sample"], sigma=50, dim=-2
    )
)

print(generator)

sample = generator[200]
print("Sample keys:", sample.keys())

fig = plt.figure(figsize=(10, 7))
axs = fig.subplots(2, 1)
axs[0].plot(sample["X"].T)
axs[1].plot(sample["y"].T)

plt.show()
