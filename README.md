
# Count Objects Until No Tomorrow (C.O.U.N.T.)
🧛‍♂️: _Ah-ah-ah!_

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![MIT License](https://img.shields.io/badge/license-MIT-green)

COUNT counts the cells in microscope videos of our DEP microfluidic devices and tells you how many went to each outlet:
- **Top outlet: pDEP** (positive DEP response)
- **Bottom outlet: nDEP** (no positive DEP response)

## What COUNT does
1. Opens your video (Nikon `.nd2` or `.avi`).
2. Finds the cells that are **moving**. Dust, cells stuck to the device and the electrodes stay still, so they are ignored.
3. Follows each moving cell through the video.
4. Counts each cell **once**, when it crosses a counting line on its way to the outlets. Cells crossing above the outlet divider are counted as pDEP; below it, nDEP.
5. Saves the counts to a spreadsheet (CSV file).

## Installing COUNT
You only need to do this once.
1. Install [Python](https://www.python.org/downloads/) (3.10 or newer).
2. Get COUNT, either:
   - with git: `git clone https://github.com/bschelske/COUNT`, or
   - by downloading it: on the [COUNT GitHub page](https://github.com/bschelske/COUNT), click the green **Code** button → **Download ZIP**, and unzip it.
3. Open a terminal in the COUNT folder (in PyCharm: the **Terminal** tab at the bottom) and run:
```
pip install -r requirements.txt
```

## Counting your videos
1. Start COUNT: open `COUNT/main.py` in PyCharm and click **Run** (or run `python COUNT/main.py` in a terminal).
2. Click **Choose File** for one video, or **Choose Folder** to count every video in a folder.
3. Click the right-hand **Choose Folder** to pick where the results are saved.
4. Click **Preview Detection** and check the settings are right for your videos (see below).
5. Click **Confirm**. Progress is shown in the PyCharm window; a video takes about a minute or less.

Not sure what a setting does? Click **Help** in the COUNT window.

## Check the preview before counting
**Preview Detection** shows two images from the middle of your video with:

| On the image | What it is | It should be... |
|---|---|---|
| **Orange circles** | Moving cells COUNT has found | on the moving cells, not on dust or electrodes |
| **Blue line** | The counting line | across the main channel, past the electrodes, before the channel splits |
| **Grey line** | The outlet divider | level with the tip that splits the two outlets |
| **Green box** | The part of the image COUNT looks at | around the channel and both outlets |

Press **4** and **5** to switch between the two images, and **Esc** to close the preview.

If the lines or box are in the wrong place (for example after moving the camera or using a different device), change these settings and preview again:
- **Counting Line x**: moves the blue line left or right (pixels from the left edge)
- **Outlet Divider y**: moves the grey line up or down (pixels from the top)
- **Region Top y** and **Region Bottom y**: the top and bottom of the green box

To keep your settings for next time, use **File → Save as default settings**. You can also keep a settings file for each setup and open it with **File → Load settings**.

The other settings rarely need changing; **Help** explains each one.

## Your results
For each video, COUNT saves two files in the results folder:
- **`<video>_results.csv`**: one row per counted cell. The **`DEP_response`** column is **True** for pDEP (top outlet) and **False** for nDEP (bottom outlet). The totals are also printed when COUNT finishes.
- **`<video>_trajectory_results.csv`**: where each counted cell was in every frame (only needed for detailed analysis).

**Double-checking a video:** tick **Save Overlay?** before clicking Confirm. COUNT then saves a picture of every frame in an `overlay` folder, showing the cells it found (orange), each cell as it is counted (green = pDEP, pink = nDEP) and the running totals.

## Plotting pDEP % against frequency
Run `COUNT/bg_plotting.py` (open it in PyCharm and click **Run**):
1. Click the first **Choose Folder** and choose your results folder. The video names need to include the frequency, for example `A375 20kHz`. This makes `final_results/final_results.csv` with the pDEP percentage for each video.
2. Click **Choose File** and choose that `final_results.csv` to plot pDEP % against frequency.

## Updating COUNT
If you installed COUNT with git, open a terminal in the COUNT folder and run:
```
git checkout master
git pull
pip install -r requirements.txt
```
If you downloaded the ZIP, download a fresh copy.

Counts from different versions of COUNT may not match, so count all videos you want to compare with the same version.

## Help and contact
- Click **Help** in the COUNT window for an explanation of every setting.
- Questions or problems: bschelske@iastate.edu, or open an issue on GitHub.
- Changing COUNT's code? See [DEVELOPERS.md](DEVELOPERS.md).
