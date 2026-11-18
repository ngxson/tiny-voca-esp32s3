# tiny-voca on the ESP32-S3

https://github.com/user-attachments/assets/30e705c4-ce2c-4228-a7ed-4b96ea30050a

Type a sentence, hear it. [tiny-voca](https://github.com/n4ze3m/tiny-voca), a 4.5M-parameter English
text-to-speech model, running entirely on an ESP32-S3: phonemizer, neural network and speaker.
No cloud, no computer once it is flashed.

## Supported boards

| Board | `BOARD` | Status |
|---|---|---|
| [Waveshare ESP32-S3-AUDIO-Board](https://www.waveshare.com/wiki/ESP32-S3-AUDIO-Board) | `waveshare_s3_audio` | Works |

## Quick start

Just one thing: a supported board and a USB-C cable. The firmware and the model are prebuilt on
[Hugging Face](https://huggingface.co/ngxson/tiny-voca-esp32s3).

```bash
pip install esptool huggingface_hub pyserial
git clone https://github.com/ngxson/tiny-voca-esp32s3 && cd tiny-voca-esp32s3
hf download ngxson/tiny-voca-esp32s3 --local-dir out
esptool write-flash 0x0 out/waveshare_s3_audio/firmware.bin 0x210000 out/tvoca.bin 0x910000 out/phonemizer.bin
./monitor.sh
```

```
tts> Hello! This is tiny voca, speaking from the little board on your desk.
```

## Commands

| Command | What it does |
|---|---|
| any text | Speak it |
| `voice F2` / `voice M2` | Female / male voice |
| `vol 50` | Speaker volume, 0 to 100 |
| `beep` | Test the speaker |
| `prof` | Print where the time goes after each sentence |
| `look 0` / `look 32` | Full attention (speech starts later) / streaming, the default |

A short sentence starts playing after about a second, a long one after a few seconds. One sentence
can be up to about 19 seconds long; longer ones are split at commas.

## Building the firmware

With [ESP-IDF](https://docs.espressif.com/projects/esp-idf/) v6.1, after the Quick start (the model
files in `out/` stay as they are):

```bash
cd firmware
. $IDF_PATH/export.sh
idf.py -B build/waveshare_s3_audio -DBOARD=waveshare_s3_audio build flash
```

Technical details, including how to export the model yourself, are in [AGENTS.md](AGENTS.md).

## License

Apache-2.0, see [LICENSE](LICENSE). `third_party/dizione` is from
[audio.cpp](https://github.com/0xShug0/audio.cpp) (Apache-2.0). The model comes from tiny-voca (Apache-2.0).
