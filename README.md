
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

See [How COUNT finds, follows and counts cells](#how-count-finds-follows-and-counts-cells) for more detail.

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

## How COUNT finds, follows and counts cells

### 1. Finding the cells that move
Our cells are small and faint, and the videos also contain dust, cells stuck to the device and the electrodes. What sets the cells we want apart is that they **move**.

So for each image, COUNT builds a picture of the background from the **10 images before and the 10 images after it**. Each pixel of the background is the middle (median) value of that pixel across those 20 images. Anything that stays in the same place for most of that time, such as dust, stuck cells and the electrodes, ends up in the background.

COUNT then compares the image with its background. Spots that are noticeably different (**Detection Threshold**, 8 grey levels) and cell-sized (between **Min** and **Max Speck Area**) are the moving cells. These are the orange circles in the preview.

This means:
- a cell that stays still for more than about half of those 20 images (about 3 seconds in our recordings) is treated as background and not found;
- very faint cells can be missed. Lowering the Detection Threshold finds fainter cells, but also more noise.

### 2. Following each cell
COUNT then links each cell from one image to the next. It uses the cell's speed so far to predict where the cell should be in the next image, and picks the closest cell to that prediction. Each cell in an image can only belong to one track, so two cells close together stay separate.

- A cell can move up to **Max Movement Between Images** (default: 100 pixels) from where it was predicted to be. In our 
  recordings, with about 3 images per second, cells move 30–100 pixels between images.
- Cells flow left to right, but a cell may pause or jitter slightly backwards (**Cell Radius**) and still be followed.
- If a cell isn't found for a few images in a row (**Timeout**), COUNT stops following it.

### 3. Counting each cell once
A cell is counted **once, when it crosses the counting line** (the blue line) moving right. It must have been seen in at least 3 images (**Min Detections to Count**), so a couple of random specks either side of the line are not counted.

Where it crosses decides its outlet:
- **above** the outlet divider (grey line): top outlet, **pDEP**;
- **below** it: bottom outlet, **nDEP**.

Cells that never reach the line, such as cells held on the electrodes, are not counted. This is why the counting line and outlet divider must be in the right place for your videos (see [Check the preview before counting](#check-the-preview-before-counting)).

### Checking COUNT against hand counts
On our test videos, COUNT's counts closely matched hand counting. On one video, for example, COUNT counted 85 nDEP / 15 pDEP against a hand count of 84 / 13. If your videos look different (another device, magnification or frame rate), count one video by hand and compare it with COUNT before relying on the results.

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
