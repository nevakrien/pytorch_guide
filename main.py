import torch

def main() -> None:
    #match follows operator order
    a = torch.eye(3)
    b = torch.ones((3, 3))
    c = torch.zeros((3, 3))
    c[0, 1] = 2.0
    c[1, 2] = 3.0
    c[2, 0] = 4.0
    print("a (eye):\n", a)
    print("b (ones):\n", b)
    print("c (zeros + values):\n", c)
    print("c@a:\n", c @ a)
    print("b*c:\n", b * c)
    y = a + b*c@a
    print("y = a + b*c@a:\n", y)
    # Operator precedence: matrix multiplication (@) happens before element-wise multiplication (*), then addition (+)

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

    #indexing by boolean lets us gather specific inputs
    #Boolean masking is also called masking - it selects elements where mask is True
    #because the amount of true vs false is unknown the output is just an array with all the hits
    x = torch.rand((5,6,7,8))
    mask = x % 2 < 0.5
    print("x:\n", x.shape)
    print("mask:\n", mask.shape)
    print("x[mask]:", x[mask].shape)

    #torch.gather selects values from one dimension using an index tensor, output shape follows the index
    #use gather when you want position-based picks with a predictable output shape
    base = torch.tensor([[10, 11, 12], [20, 21, 22]])
    idx = torch.tensor([[2, 0], [1, 1]])
    gathered = torch.gather(base, 1, idx)
    print("base:\n", base)
    print("idx:\n", idx)
    print("gathered (torch.gather):\n", gathered)
    #Summary: use boolean masks for filtering by condition, use gather for position-based picks

    #integer tensor indexing can replicate gather by building index tensors for all dims you want to pick
    #use integer indexing when you need more flexible multi-dimension indexing patterns
    row_idx = torch.arange(base.size(0)).unsqueeze(1).expand_as(idx)
    indexed = base[row_idx, idx]
    print("base:\n", base)
    print("idx:\n", idx)
    print("indexed (int tensor):\n", indexed)


if __name__ == "__main__":
    main()
