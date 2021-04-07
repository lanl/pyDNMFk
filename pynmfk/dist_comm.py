# @author: Manish Bhattarai
class MPI_comm():
    """Initialization of MPI communicator to construct the cartesian topology and sub communicators

    Parameters
    ----------
    comm : object
        MPI communicator object
    p_r : int
        row processors count
    p_c : int
        column processors count"""


    # MPI Initialization here
    def __init__(self, comm, p_r, p_c):
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.p_r = p_r
        self.p_c = p_c
        self.cartesian2d = self.comm.Create_cart(dims=[self.p_r, self.p_c], periods=[False, False], reorder=False)
        self.coord2d = self.cartesian2d.Get_coords(self.rank)

    def cart_1d_row(self):
        """
        Constructs a cartesian row communicator through construction of a sub communicator across rows

        Returns
        -------
        cartesian1d_row : object
            Sub Communicator object
        """
        self.cartesian1d_row = self.cartesian2d.Sub(remain_dims=[True, False])
        self.rank1d_row = self.cartesian1d_row.Get_rank()
        self.coord1d_row = self.cartesian1d_row.Get_coords(self.rank1d_row)
        return self.cartesian1d_row

    def cart_1d_column(self):
        """
        Constructs a cartesian column communicator through construction of a sub communicator across columns

        Returns
        -------
        cartesian1d_column : object
            Sub Communicator object
        """
        self.cartesian1d_column = self.cartesian2d.Sub(remain_dims=[False, True])
        self.rank1d_column = self.cartesian1d_column.Get_rank()
        self.coord1d_column = self.cartesian1d_column.Get_coords(self.rank1d_column)
        return self.cartesian1d_column

    def Free(self):
        """ Frees the sub communicators"""
        self.cart_1d_row().Free()
        self.cart_1d_column().Free()
