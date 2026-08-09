# Data layout

The experiment configurations use the following default layout:

```text
data/
├── cifar100/
├── imagenet200/
│   ├── train/<class-id>/*.JPEG
│   └── val/<class-id>/*.JPEG
├── device_sweeps/
└── language/
    ├── gpt2/
    │   ├── tokenizer.json
    │   ├── vocab.json
    │   └── merges.txt
    ├── openwebtext/
    │   ├── owt_train.txt
    │   └── owt_valid.txt
    └── benchmarks/
        ├── tinystories_valid.txt
        └── blimp/
```

Download CIFAR datasets:

```bash
python data/prepare_cifar.py --dataset all
```

Prepare OpenWebText:

```bash
python data/prepare_openwebtext.py \
  --output-dir data/language/openwebtext
```

Prepare TinyStories validation data:

```bash
python data/prepare_tinystories.py \
  --output data/language/benchmarks/tinystories_valid.txt
```

ImageNet images are not redistributed. Arrange the selected 200 classes as
`train/<class-id>` and `val/<class-id>` directories before launching the
ImageNet-200 matrix.
