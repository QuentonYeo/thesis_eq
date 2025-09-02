import seisbench
import seisbench.data as sbd
import os
import matplotlib.pyplot as plt

data = sbd.DummyDataset()
print(data)

print("Cache root:", seisbench.cache_root)
print("Contents:", os.listdir(seisbench.cache_root))
print("datasets:", os.listdir(seisbench.cache_root / "datasets"))
print("dummydataset:", os.listdir(seisbench.cache_root / "datasets" / "dummydataset"))

dummy_from_disk = sbd.WaveformDataset(
    seisbench.cache_root / "datasets" / "dummydataset"
)
print(dummy_from_disk)
print(data.metadata)

waveforms = data.get_waveforms(3)
print("waveforms.shape:", waveforms.shape)

plt.plot(waveforms.T)

# # Filtering the dataset
# mask = (
#     data.metadata["source_magnitude"] > 2.5
# )  # Only select events with magnitude above 2.5
# data.filter(mask)

# print(data)
# print(data.metadata)


magnitudes = data.metadata["source_magnitude"]
indicies = data.metadata["index"]

print(indicies)


# Plot histogram: frequency vs magnitude bins (0-10, step 0.5)
bins = [x * 1 for x in range(11)]  # 0, 0.5, ..., 10
plt.figure()
plt.hist(magnitudes, bins=bins, edgecolor="black")
plt.xlabel("Magnitude")
plt.ylabel("Frequency")
plt.title("Magnitude Distribution in DummyDataset")
plt.xticks(bins)
plt.show()
