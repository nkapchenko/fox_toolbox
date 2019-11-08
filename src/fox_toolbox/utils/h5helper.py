import tables
import uuid
import numpy as np
import pandas as pd
import requests


def get_tmp_name():
    """
    Returns a dummy random name
    """
    return str(uuid.uuid4())


def read_file_to_store(file_path):
    """
    :return: HDF5Store object
    """
    return tables.open_file(file_path, driver="H5FD_CORE")


def read_url_to_store(url, session=None):
    """
    Reads content of an url into in-memory hdf store of tables format
    :param url: url link to h5 file
    :param session: session object or None to create a new one
    :return: HDF5Store object
    """
    if session is None:
        session = requests.Session()
    with session.get(url) as response:
        content = response.content
        hdfstore = tables.open_file(get_tmp_name(), driver="H5FD_CORE",
                                    driver_core_backing_store=0,
                                    driver_core_image=content)
        return hdfstore


def get_node_type(node):
    """
    Get type of node
    :param node: node of h5file
    :return: node type name or None
    """
    if 'type' in node._v_attrs:
        return node._v_attrs.type.decode('UTF-8')
    return None


def parse_grid(hdf, grid_group):
    """
    Parses node group of nD_grid type into a flattened table.
    :param hdf: HDF5Store object
    :param grid_group: Grid group node
    :return: tuple(axes_names, data) where axes_names is a tuple of
        axis column names in order of appearance; data is a dictionary where
        key is name of the data and value is np.array of values
    """

    def get_array_nodes(group):
        return hdf.list_nodes(group, classname='Array')

    def parse_arr_node(arr_node):
        """
        For only 1 scenario array(0.0) is treated as empty array object.
         Function converts it back to array of size 1.
        """
        res = arr_node.read()
        node_name = arr_node.name
        # 1 zero element is treated as non-array
        return node_name, (res.reshape(1) if len(res.shape) == 0 else res)

    n_type = get_node_type(grid_group)
    if not n_type or 'D_grid' not in n_type:
        raise Exception('Unexpected grid type: %s' % n_type)

    # parse AxisX, AxisY, ...
    data = {}
    for ax in grid_group:
        axis_name = ax._v_name
        if 'Axis' in axis_name:
            name, vals = parse_arr_node(get_array_nodes(ax)[0])
            if name in data:
                name += axis_name[-1]
            data[name] = vals

    # 'flatten' N axes into table
    axes_names = tuple(data.keys())
    axes_mesh = np.meshgrid(*data.values(), indexing='ij')
    for k, v in zip(axes_names, axes_mesh):
        data[k] = v.flatten('C')
    # parse values
    data.update(map(parse_arr_node, get_array_nodes(grid_group.Values)))
    return axes_names, data


def parse_grid_to_df(hdf, grid_group):
    """
    Parses node group of nD_grid type to a DataFrame.
    :param hdf: HDF5Store object
    :param grid_group: Grid group node
    :return: DataFrame with all node information where:
        df.name is the name of grid_group node
        df.index is index based on grid axes
    """
    grid_name = grid_group._v_pathname # grid_group._v_name
    try:
        axes_names, data = parse_grid(hdf, grid_group)
        df = pd.DataFrame(data)
        df.name = grid_name
        df.set_index(list(axes_names), inplace=True)
        return df
    except ValueError:
        print(f'Error occurred during parsing of {grid_name}')
        raise


def parse_all_grids(hdf):
    """
    Yields dataframes for each group of nD_grid type.
    :param hdf: HDF5Store object
    :return: collection of parsed dataframes
    """
    for n in hdf.walk_groups():
        n_type = get_node_type(n)
        if n_type and 'D_grid' in n_type:
            yield parse_grid_to_df(hdf, n)
