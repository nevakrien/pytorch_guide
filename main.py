import torch
import warnings

#look at main below for runing specific things

def tensor_basics() -> None:
    # tensor = n-dimensional array (like numpy), holds numbers + shape + dtype
    # arange makes a 1D tensor with evenly spaced values
    x = torch.arange(6)
    print("x (arange):", x)

    # reshape changes the view of data without copying when possible
    x2 = x.reshape(2, 3)
    print("x reshaped to 2x3:\n", x2)

    # rand creates random floats in [0, 1)
    r = torch.rand((2, 3))
    print("r (rand):\n", r)

    # randn creates random floats from a normal distribution (mean=0, std=1)
    n = torch.randn((2, 3))
    print("n (randn):\n", n)

    # zeros/ones create tensors filled with 0 or 1
    z = torch.zeros((2, 3))
    o = torch.ones((2, 3))
    print("z (zeros):\n", z)
    print("o (ones):\n", o)

    # permute reorders dimensions (like transpose for more than 2 dims)
    p = torch.rand((2, 3, 4))
    print("p.shape:", p.shape)
    print("p.permute(1, 0, 2).shape:", p.permute(1, 0, 2).shape)

    # tensor() wraps a Python list and sets dtype if needed
    t = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    print("t (tensor from list):\n", t)
    print("t.shape:", t.shape, "t.dtype:", t.dtype)


def order_of_ops() -> None:
    # basic arithmetic with tensors
    a = torch.eye(3) #identity matrix
    b = torch.ones((3, 3))
    c = torch.zeros((3, 3))
    c[0, 1] = 2.0
    c[1, 2] = 3.0
    c[2, 0] = 4.0
    print("a (eye):\n", a)
    print("b (ones):\n", b)
    print("c (zeros + values):\n", c)
    print("c@a (matmul):\n", c @ a)
    print("b*c (elementwise):\n", b * c)
    y = a + b*c@a
    print("y = a + b*c@a:\n", y)
    # Operator precedence: matrix multiplication (@) happens before element-wise multiplication (*), then addition (+)

def gradients_and_autograd() -> None:
    # autograd builds a computation graph as you do tensor ops
    # all tensors with requires_grad=True track gradients
    x = torch.tensor(2.0, requires_grad=True)
    y = x**2 + 3 * x
    print("y:", y)

    # backward computes dy/dx and stores it in x.grad
    y.backward()
    print("x.grad (dy/dx):", x.grad)

    # we can zero gradient like so
    # usually you would use .zero_grad() on a model/optimizer instead
    x.grad.zero_()
    print("x.grad after zeroing:", x.grad)


    # we can turn off tracking like so
    # operations under no_grad() won't build a graph
    with torch.no_grad():
        y = x**2 + 3 * x
    print("y.requires_grad under no_grad:", y.requires_grad)

    # build a new tracked graph to show backward still works
    y = x
    y.backward()
    print("x.grad after fresh backward:", x.grad)

    # higher-order grads: create_graph=True keeps graph for gradients
    x2 = torch.tensor(2.0, requires_grad=True)
    y2 = x2**3
    grad1 = torch.autograd.grad(y2, x2, create_graph=True)[0]
    grad2 = torch.autograd.grad(grad1, x2)[0]
    print("first derivative (dy/dx):", grad1)
    print("second derivative (d2y/dx2):", grad2)


def stack_and_concat() -> None:
    # stack adds a new dimension, cat joins along an existing one
    # stack inputs are tensors, so it DOES keep gradients
    first = torch.tensor([1.0, 2.0], requires_grad=True)
    second = torch.tensor([3.0, 4.0], requires_grad=True)

    # stack takes a list of same-shaped tensors and creates a new dimension
    # result shape here is (2, 2) because we added a dimension at dim=0
    stacked = torch.stack([first, second], dim=0)

    # cat (concat) joins tensors along an existing dimension
    # result shape here is (4,) because it stitches dim=0 together
    concatenated = torch.cat([first, second], dim=0)

    print("stacked shape:", stacked.shape)
    print("concatenated shape:", concatenated.shape)
    print("stacked:\n", stacked)
    print("concatenated:\n", concatenated)

    stacked.sum().backward()
    print("first.grad after stack:", first.grad)
    print("second.grad after stack:", second.grad)


def breaking_gradients() -> None:
    #it is generally fairly hard to break gradients
    #you would most likely get some sort of warning or error if you do

    source = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    doubled = source * 2

    # detach is an explicit way to cut the gradient
    # it produces a tesnor of the same values but a fresh gradient
    # operations on detach would not go back to the original
    detached = doubled.detach()

    print("detached.requires_grad:", detached.requires_grad)
    try:
        detached.sum().backward()
    except RuntimeError as error:
        print("backward on detached tensor:", error)

    source.grad = None
    doubled.sum().backward()
    print("source.grad after normal backward:", source.grad)

    #because numpy cant track gradients
    #pytorch wont let us call numpy() on a tesnor that has gradients
    try:
        doubled_numpy = doubled.numpy()
        print("doubled.numpy() result:", doubled_numpy)
    except RuntimeError as error:
        print("doubled.numpy() error:", error)

    #instead we need to EXPLICTLY remove the gradient
    #this lets us know we have lost gradient privlages
    #if the tensor was on GPU we would also need to move to cpu
    detached_numpy = doubled.detach().cpu().numpy()
    print("to_numpy(doubled):", detached_numpy)


def unsqueeze_expand() -> None:
    # unsqueeze adds a size-1 dimension (useful before expand/broadcast)
    # expand repeats along size-1 dims without copying data
    # expand_as uses another tensor's shape instead of explicit sizes
    
    example = torch.arange(3)# 1 2 3 similar to range
    print("example:", example)
    print("example.unsqueeze(1).shape:", example.unsqueeze(1).shape)

    #we can do unsqueeze in a nice syntax like so
    print("example[:,None].shape:", example[:,None].shape)


    print("example.unsqueeze(1).expand(3, 2):\n", example.unsqueeze(1).expand(3, 2))
    target = torch.zeros((3, 2))
    print("example.unsqueeze(1).expand_as(target):\n", example.unsqueeze(1).expand_as(target))


def broadcast_matmul() -> None:
    # broadcasting matrix mul
    # there is [5 7 8] [8 9]
    # the 2 8s have to be the same number
    # what happens is that each [7 8] of the 5 matricies in left
    # gets multiplied by each
    left = torch.rand((5, 7, 8))
    right = torch.rand((8, 9))
    out = left @ right
    print("left.shape:", left.shape)
    print("right.shape:", right.shape)
    print("out.shape:", out.shape)


def reductions() -> None:
    # reductions: mean/max/median work as torch.fn(x) or x.fn()
    stats = torch.arange(12).reshape(3, 4).float()
    print("stats:\n", stats)
    print("torch.mean(stats):", torch.mean(stats))
    print("stats.mean():", stats.mean())

    # dim tells which axis to reduce (dim=0 columns, dim=1 rows for 2D)
    print("stats.mean(dim=1):", stats.mean(dim=1))
    print("stats.mean(dim=0):", stats.mean(dim=0))

    # keepdim keeps the reduced dimension as size-1 so shapes still line up
    print("stats.mean(dim=1, keepdim=True):\n", stats.mean(dim=1, keepdim=True))

    # std stands for standard deviation (how spread out the values are)
    # default is the sample version: divide by n-1 for a better estimate
    # use unbiased=False for population std if you need it
    print("stats.std(dim=1):", stats.std(dim=1))
    print("stats.std(dim=1, unbiased=True):", stats.std(dim=1, unbiased=True))
    print("stats.std(dim=1, unbiased=False):", stats.std(dim=1, unbiased=False))

    # max/median works similarly but we have not only values but also indexes
    max_vals, max_idx = stats.max(dim=0)
    print("stats.max(dim=0):", max_vals, "indices:", max_idx)
    median_vals, median_idx = stats.median(dim=1)
    print("stats.median(dim=1):", median_vals, "indices:", median_idx)

def simple_indexing() -> None:
    x = torch.arange(24).reshape(2, 3, 4)
    print("x shape:", x.shape)

    print("x[0].shape:", x[0].shape)        # drop leading dim
    print("x[:, 1].shape:", x[:, 1].shape)  # select along dim
    print("x[..., 2].shape:", x[..., 2].shape)

    # None / unsqueeze
    # sometimes we wana add a dimention
    # Nones get replaced with just an additional new dimention of size 1 same as np.newaxis
    print("x[:, None].shape:", x[:, None].shape)
    print("x.unsqueeze(1).shape:", x.unsqueeze(1).shape)  # same as x[:, None]

    # Mixing slice + integer
    print("x[1, :, 0]:\n", x[1, :, 0])


def boolean_indexing() -> None:
    # indexing by boolean lets us gather specific inputs
    # Boolean masking is also called masking - it selects elements where mask is True
    # because the amount of true vs false is unknown the output is just an array with all the hits
    x = torch.rand((5,6,7,8))
    mask = x % 2 < 0.5 #boolean tensor
    print("x:\n", x.shape)
    print("mask:\n", mask.shape)
    print("x[mask]:", x[mask].shape)

def integer_indexing() -> None:
    # Integer tensor indexing (a.k.a. advanced indexing)
    # Index tensors are broadcast together
    # Output shape == broadcasted index shape (+ remaining dims)

    x = torch.arange(9).reshape(3, 3)
    print("x:\n", x)

    idx = torch.tensor([[1, 1],
                        [2, 2]])

    y = x[idx]
    print("idx shape:", idx.shape)
    print("x[idx] shape:", y.shape)
    print("x[idx]:\n", y)

def gather_examples() -> None:
    # torch.gather selects values from one dimension using an index tensor, output shape follows the index
    # use gather when you want position-based picks with a predictable output shape
    base = torch.tensor([[10, 11, 12], [20, 21, 22]])
    idx = torch.tensor([[2, 0], [1, 1]])
    gathered = torch.gather(base, 1, idx)
    print("base:\n", base)
    print("idx:\n", idx)
    print("gathered (torch.gather):\n", gathered)
    # Summary: use boolean masks for filtering by condition, use gather for position-based picks

    # integer tensor indexing can replicate gather by building index tensors for all dims you want to pick
    # use integer indexing when you need more flexible multi-dimension indexing patterns
    row_idx = torch.arange(base.size(0)).unsqueeze(1).expand_as(idx)
    indexed = base[row_idx, idx]
    print("base:\n", base)
    print("idx:\n", idx)
    print("indexed (int tensor):\n", indexed)


#this one i barely seen but it can be usefull
#its mostly for advanced users
def einsum_examples() -> None:
    # einsum uses index notation (i,j,k) to describe tensor ops succinctly
    # matrix multiply: (i,j) x (j,k) -> (i,k)
    a = torch.rand((2, 3))
    b = torch.rand((3, 4))
    mm = torch.einsum("ij,jk->ik", a, b)
    print("einsum matmul shape:", mm.shape)

    # batch matmul: (b,i,j) x (b,j,k) -> (b,i,k)
    left = torch.rand((5, 2, 3))
    right = torch.rand((5, 3, 4))
    bmm = torch.einsum("bij,bjk->bik", left, right)
    print("einsum batch matmul shape:", bmm.shape)

    # sum over a dimension: (i,j) -> (i)
    summed = torch.einsum("ij->i", a)
    print("einsum sum over j:", summed)




if __name__ == "__main__":
    # you uncomment stuff you care about
    tensor_basics()
    # order_of_ops()
    # gradients_and_autograd()
    # stack_and_concat()
    # breaking_gradients()
    # unsqueeze_expand()
    # broadcast_matmul()
    # reductions()
    # boolean_indexing()
    # simple_indexing()
    # integer_indexing()
    # gather_examples()
    # einsum_examples()
