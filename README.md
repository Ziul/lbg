# Vector Quantization Coding
Encode and decode a file using the Linde-Buzo-Gray algorithm


### Install

```
    $ pip install -e .
```
  
### Using

Examples:

Build codebooks with some pack of images:
```
    lbg-learn <PATH> [FLAGS]
```

Quantize a file using the codebook from the deep learning:
```
    lbg <file>
```

Test a file with a custom quantization:
```
    lbg-test <file> [FLAGS]
```
    
  
More options:

```
    $ lbg --help
```
