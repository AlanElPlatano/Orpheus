# Orpheus - A basic songwriter AI

This project is a basic songwriter AI, aimed at producing MIDI files with melody tracks and chord (harmony) tracks, intended to be used as a simple songwriting tool for musicians, producers, and (of course) songwriters.

This AI is supposed to take MIDI files in their most basic structure, being a monophonic melody track and a chord track with no rhythmic variation, with chords being sustained all the way until the next one starts. For proper function of the AI, your training files will have to look something like this:

![Image showcasing the basic melody and chord track structure detailed above](assets/partiture_sample.png "Screenshot from Guitar Pro")

This structure will make sure the AI learns the basic structure of songs (chords and melody), allowing it to train on the core aspects parts of your song dataset, after this, you can manually add more production (chord rhythmic variations, more instruments, bass, etc) to your own liking.

### How does the AI Identify Tracks In My Dataset?

The AI uses a 3-layered system to automatically identify melody and chord tracks:

1. **Track Name Recognition**: Detects keywords like "melody", "lead", "solo", "voice" for melodies, and "chord", "accomp", "harmony", "pad" for chords (supports multiple languages)

2. **Polyphony Analysis** (Primary Method):
   - **Melody tracks**: Monophonic or light polyphony (≤2 notes playing simultaneously)
   - **Chord tracks**: High polyphony (≥3 notes playing simultaneously)

3. **Instrument Analysis**: Uses MIDI program numbers and channel information

The system uses all three methods for accurate classification. This means you can name your tracks in your preferred language, or let the AI analyze the musical content automatically.

### Note Range and Constraints

- **Bass Track Removal**: Tracks with more than 35% of notes below C2 (MIDI note 36) are automatically removed during preprocessing.
- **Training Pitch Range**: C2 to C6 (MIDI notes 36-84) - optimized for vocal and guitar melodies
- This range covers most melodic and harmonic content while filtering out bass-heavy material 

## Installation

First, make sure you have at least python 3.10 installed, you can check your version by executing this in your console:

```
python -V
```

I recommend using python 3.10.8 as this is the version i had installed when developing the project, but all newer versions of python should work.

---

Then, clone the repo either downloading the zip from Github or by cloning using git (recommended):

```
git clone https://github.com/AlanElPlatano/Orpheus.git
```

---

Then, we have to create a python virtual environment to handle all the libraries used (strongly recommended), for this make sure you have your VSCode console on the root of your project and type:
```
python -m venv my_venv
```
Where 'my_venv' is the name of your virtual environment, you can use whatever name you want. This will create a folder with that same name on your project root. Now we have to activate it with the following command:
```
{name_of_your_venv}/Scripts/Activate.ps1
```

This will make the 'venv' activate, you can know this worked because a prefix with the name of the virtual environment like (venv) or (my_venv) will show up before every command in your console.

---

Now we have to install the libraries. First, PyTorch (the library that allows our AI to work) requires special installation steps depending on your hardware.

#### For NVIDIA GPU Users (Recommended for faster training)

If you have an NVIDIA graphics card with CUDA support:

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

This installs PyTorch with CUDA 11.8 support for GPU acceleration. **Note:** This will download approximately 3GB, so be patient.

#### For CPU-Only Systems (AMD GPUs, Intel, or no dedicated GPU)

If you don't have an NVIDIA GPU or want CPU-only installation:

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Note:** CPU training will be significantly slower but fully functional.

#### For Other CUDA Versions

If you have a different CUDA version installed, check the [PyTorch website](https://pytorch.org/get-started/locally/) to get the correct installation command for your system.

---

After installing PyTorch, install the remaining dependencies:

```
pip install -r requirements.txt
```

This will install all other required libraries (Gradio, Pretty MIDI, MidiTok, etc.). This may take a few minutes, but this should be a much faster download than PyTorch.


## Usage

First, copy all of your files into the 'source_midis' folder on the root of the project, here is where the processing will start. Then, open the gradio GUI by activating the following command (remember to have your venv activated):
```
python -m gui.gradio_app
```

After about 10 seconds, you will see a message like this one on the console:
```
* Running on local URL:  http://localhost:42069
* To create a public link, set `share=True` in `launch()`.
```

With this, we have our GUI ready, you can open any browser (Chrome, Edge, Firefox, Opera, etc) and go to 'http://localhost:42069', here you will see the Gradio GUI:

![Screenshot of the Preprocess tab of the Gradio GUI](assets/preprocess_gui.png "Browser screenshot")

The GUI is comprised of several tabs that each pertain to a specific step in our AI pipeline. The first one we see is the 'Preprocess' tab.

In this tab we have several optional tools to preprocess the files we placed in our 'source_midis' folder. Before we permanently modify our MIDIs, on the top of the left column we have 'Backup & Recovery', which can backup our MIDIs before the preprocessor touches them, as well as recovering them. You can use this option if you want to experiment with your MIDIs before moving on to the next tab. 

In the middle column we have the preprocessing options themselves:
- A quantization to align notes to the grid (recommended)
- General note cleanup
- Empty track cleanup
- Remove tracks with bass content (if more than 35% of notes are below the threshold, the entire track is removed)

On the right we have the Process button and a status log for additional clarity. The processed files will overwrite the original ones, hence why the backup function is provided.

---

### Then, we have the MIDI Parser tab:

![Screenshot of the MIDI Parser tab of the Gradio GUI](assets/midi_parser.png "Browser screenshot")

Our AI cannot understand MIDI files because they're binaries, so this tab processes them into a JSON format which is easier to understand. Here we only have 2 settings:

- Simple/Advanced Mode: it determines the way that the parsing is done, by now only the simple mode is supported for future steps so don't parse your files with Advanced mode (development pending).

- Compress JSON, this option is only meant if you plan to use the parsed JSONs for something other than this AI (like sharing them or storing them). If you are going to use them for training leave this option unchecked

On the bottom we get the basic buttons to start the process and cancel it. When you finish parsing, you will see a log on screen like this one:

![Console screenshot of the MIDI Parser after processing the entire folder](assets/parser_console.png "Console screenshot")

**Note:** The parser has a limit of 2048 tokens per file so very long files will be truncated to fit within this limit. If you get any truncated files i **strongly** recommend to import that file into any MIDI editor and split it into several smaller files to make sure all your dataset gets used for training and nothing gets discarded.

---

### Up next, we have the Augmentation tab:

![Screenshot of the Augmentation tab](assets/augmentation_tab.png "Augmentation tab")

This is a very special tab, it takes every parsed file and transposes it into all 12 keys, by creating duplicates of each song from -6 semitones to +5 semitones (files are not tranposed from minor to major keys and viceversa), we do this to accomplish 2 things:

- By having songs in all 12 keys, we make sure that the model learns scales (music theory) intrinsically, if your dataset favored certain keys, the generation quality for those keys would be better than for all the other keys. By exploding every file into all 12 keys, we make sure that every song is present in all pitches, which allows the AI to learn all 12 pitches.

- To have 12 times as much training dataset, although not as good as an ideal case where we have a huge training dataset in equally distributed files, it still helps us have more keys. The scripts have a way to avoid the model overfitting (memorizing instead of understading) on our data by splitting the dataset in 3 groups (training, testing, valoration) where all 12 versions of each song are always placed in the same group.

---

### The Training Tab

![Screenshot of the Training tab of the Gradio GUI](assets/training_tab.png "Training Tab")

Here is one of the most important tabs in our project, here we will train our models and will be where we will spend most of our time.

The GUI provides multiple parameters to train our AI, all the way from Batch Size, Learning Rate, Number of Epochs, Warmup Steps and even an Advanced Settings menu.

For easier use, a preset menu is provided, where we have 4 options:

- **default:** This is the default training option, in this preset we're making a full run of the training process with modest quality.
- **quick_test:** This is a preset meant to verify that everything up to this point (data preparation, JSON parsing, etc) has worked correctly. This preset will only train once so you can use it before going to another option to verify everything works correctly.
- **overfit:** This is a diagnostic preset that purposefully overfits on a small batch. While overfitting (memorizing instead of understanding) is normally undesired, this preset is useful to verify your AI pipeline is working correctly, if the model CAN'T memorize a small batch, something is broken in your code. Use this as a sanity check after making changes or when debugging training issues, this a feature for developers and not end users.
- **production (recommended):** This is the best training preset, it will make a long training process that will take several hours to finish training and will produce a model ready for production. Make sure you have a decent dataset (several thousand files after augmentation) before using this preset

You can select a preset by opening the drop down menu and clicking 'Load Preset', this will replace all the options with the ones from the preset. Once you have everything ready, you can scroll down and click 'Start Training' which will start the process.

**Important Note:** The first time you start training, the system will automatically split your augmented dataset into training (80%), validation (10%), and test (10%) sets. This ensures all 12 transpositions of each song stay together to prevent data leakage. You'll see a message confirming the split was generated before training begins.

Unless you selected 'quick_test' this will start a process that will mostly take several hours to finish, for reference, my RTX 3060 took about 12 hours with the production preset and 3000 files.

#### Pause and Resume During Training

If you need to briefly pause training (for example, to free up the GPU for another task), use the **⏸️ Pause** button. This will temporarily halt training while keeping everything in memory. Click **▶️ Resume** to continue from where you left off.

**Important:** The Pause/Resume feature only works within the current session. If you click **🛑 Stop** or close the GUI, you'll need to resume from a checkpoint

### Checkpoint System

Because this process is very long, i have implemented a checkpoint system:

![Screenshot of the Training tab of the Gradio GUI, scrolled down to show the 'Checkpoint' options](assets/checkpoints_system.png "Checkpoints in the Training Tab")

During training, checkpoints are automatically saved **every 2000 steps** (or every 50 steps for quick_test, 1000 for overfit). The system keeps the 5 most recent checkpoints and automatically deletes older ones to save disk space. Each checkpoint contains your model's current state, optimizer settings, and training progress.

In this menu you can find several options, first we have a "Select Checkpoint" menu that lists all the current checkpoints, you can select one from the list and then click "Load Checkpoint", it doesn't show up on the list you can try the "Refresh List" button. Below it you can find a status windows along with another window that lists all the checkpoints in a non-interactive list, along with its own List and Refresh buttons.

Once you finish training, we can move to the next tab.

---

### The Generator tab

![Screenshot of the Generator tab of the Gradio GUI](assets/generator_tab.png "Generator Tab")

This is the tab where we will generate our files with the models we've trained. On the top left you can select from all the models you have, all the checkpoints created from the last model you trained will also show up here.

We also get generation settings, where we can choose between three modes:

- **Quality (Conservative)**: Uses Temperature 0.8, Top-p 0.95, Repetition Penalty 1.1. Stays close to training data for predictable, safe outputs.
- **Creative (Experimental)**: Uses Temperature 1.1, Top-p 0.92, Repetition Penalty 1.05. Produces more varied, experimental outputs with higher randomness.
- **Custom**: Allows you to manually adjust the sampling parameters to your preference.

Below it we get options for conditional generation, this feature is not implemented yet so it doesn't have an impact on the generated files.

And at the bottom we get some output options, where we can define the number of generated files and max length. Right below these options we get the 'Generated Music', which will use the model to generate some files, this process should only take a few minutes depending on how many files you are generating.

If everything up to this point went well, you should have your generated MIDI files in the output path. You can import these files into any MIDI software (Reaper, Guitar Pro, Sybellius, Tux Guitar, Pro Tools, etc) and you will have your music. Enjoy!
