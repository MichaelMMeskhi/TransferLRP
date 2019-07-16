# TransferLRP

Decomposing the Layer-wise Relevance Propagation down to the granular level (neuron level). We managed to determine which neurons are most relevant by calculating a simple threshold `noderel()` for them. There are various ways to implement a threshold equation depending on given task and data. 

To run the experiment do the following:

```bash
$ pip3 install -r requirements.txt
```
After installing all the necessary dependencies. Run the experiment in `run.py`:

```bash
$ python3 run.py
```
This will generate `.png` files in the `/results/` directory along with a `nodeRel.csv` file that specifies the most relevant neuron indices.

* Please read the `docstring` in `run.py` to modify the experiment to your needs.