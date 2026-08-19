"""Private runtime references for the beginner-first broadcasting workbook."""

from __future__ import annotations

import torch

CASES = {'ex001': {'expression': 'a + b',
           'intermediates': {},
           'audit': ['a', 'b'],
           'value_exprs': {},
           'expected_compatible': True,
           'mode': 'basic'},
 'ex002': {'expression': 'a + b',
           'intermediates': {},
           'audit': ['a', 'b'],
           'value_exprs': {},
           'expected_compatible': True,
           'mode': 'basic'},
 'ex003': {'expression': 'a * b',
           'intermediates': {},
           'audit': ['a', 'b'],
           'value_exprs': {},
           'expected_compatible': True,
           'mode': 'basic'},
 'ex004': {'expression': 'a + b',
           'intermediates': {},
           'audit': ['a', 'b'],
           'value_exprs': {},
           'expected_compatible': True,
           'mode': 'basic'},
 'ex005': {'expression': 'a * b',
           'intermediates': {},
           'audit': ['a', 'b'],
           'value_exprs': {},
           'expected_compatible': True,
           'mode': 'basic'},
 'ex006': {'expression': 'a - b',
           'intermediates': {},
           'audit': ['a', 'b'],
           'value_exprs': {},
           'expected_compatible': True,
           'mode': 'basic'},
 'ex007': {'expression': 'a + b',
           'intermediates': {},
           'audit': ['a', 'b'],
           'value_exprs': {},
           'expected_compatible': True,
           'mode': 'basic'},
 'ex008': {'expression': 'a + b',
           'intermediates': {},
           'audit': ['a', 'b'],
           'value_exprs': {},
           'expected_compatible': True,
           'mode': 'basic'},
 'ex009': {'expression': 'a + b',
           'intermediates': {},
           'audit': ['a', 'b'],
           'value_exprs': {},
           'expected_compatible': True,
           'mode': 'basic'},
 'ex010': {'expression': 'a * b',
           'intermediates': {},
           'audit': ['a', 'b'],
           'value_exprs': {},
           'expected_compatible': True,
           'mode': 'basic'},
 'ex011': {'expression': 'a + b',
           'intermediates': {},
           'audit': ['a', 'b'],
           'value_exprs': {},
           'expected_compatible': True,
           'mode': 'basic'},
 'ex012': {'expression': 'a - b',
           'intermediates': {},
           'audit': ['a', 'b'],
           'value_exprs': {},
           'expected_compatible': True,
           'mode': 'basic'},
 'ex013': {'expression': 'a + b',
           'intermediates': {},
           'audit': ['a', 'b'],
           'value_exprs': {},
           'expected_compatible': True,
           'mode': 'basic'},
 'ex014': {'expression': 'a * b',
           'intermediates': {},
           'audit': ['a', 'b'],
           'value_exprs': {},
           'expected_compatible': True,
           'mode': 'basic'},
 'ex015': {'expression': 'a + b',
           'intermediates': {},
           'audit': ['a', 'b'],
           'value_exprs': {},
           'expected_compatible': True,
           'mode': 'basic'},
 'ex016': {'expression': 'a + b',
           'intermediates': {},
           'audit': ['a', 'b'],
           'value_exprs': {},
           'expected_compatible': True,
           'mode': 'basic'},
 'ex017': {'expression': 'matrix + row',
           'intermediates': {},
           'audit': ['matrix', 'row'],
           'value_exprs': {},
           'expected_compatible': True,
           'mode': 'basic'},
 'ex018': {'expression': 'matrix + column',
           'intermediates': {},
           'audit': ['matrix', 'column'],
           'value_exprs': {},
           'expected_compatible': True,
           'mode': 'basic'},
 'ex019': {'expression': 'column + row',
           'intermediates': {},
           'audit': ['column', 'row'],
           'value_exprs': {},
           'expected_compatible': True,
           'mode': 'basic'},
 'ex020': {'expression': 'rows_col * grid',
           'intermediates': {'rows_col': 'rows[:, None]'},
           'audit': ['rows', 'grid'],
           'value_exprs': {},
           'expected_compatible': True,
           'mode': 'result'},
 'ex021': {'expression': 'cols_row + grid',
           'intermediates': {'cols_row': 'cols[None, :]'},
           'audit': ['cols', 'grid'],
           'value_exprs': {},
           'expected_compatible': True,
           'mode': 'result'},
 'ex022': {'expression': 'prepared + grid',
           'intermediates': {'prepared': 'values.unsqueeze(0)'},
           'audit': ['values', 'grid'],
           'value_exprs': {},
           'expected_compatible': True,
           'mode': 'result'},
 'ex023': {'expression': 'prepared * grid',
           'intermediates': {'prepared': 'values.unsqueeze(1)'},
           'audit': ['values', 'grid'],
           'value_exprs': {},
           'expected_compatible': True,
           'mode': 'result'},
 'ex024': {'expression': 'channel_view + images',
           'intermediates': {'channel_view': 'channels[None, :, None, None]'},
           'audit': ['channels', 'images'],
           'value_exprs': {},
           'expected_compatible': True,
           'mode': 'result'},
 'ex025': {'expression': 'matrix + vector',
           'intermediates': {},
           'audit': ['matrix', 'vector'],
           'value_exprs': {},
           'expected_compatible': True,
           'mode': 'aligned'},
 'ex026': {'expression': 'images - background',
           'intermediates': {},
           'audit': ['images', 'background'],
           'value_exprs': {},
           'expected_compatible': True,
           'mode': 'aligned'},
 'ex027': {'expression': 'a + b',
           'intermediates': {},
           'audit': ['a', 'b'],
           'value_exprs': {},
           'expected_compatible': True,
           'mode': 'aligned'},
 'ex028': {'expression': 'prepared + images',
           'intermediates': {'prepared': 'per_image_width_offset[:, None, :]'},
           'audit': ['images', 'per_image_width_offset'],
           'value_exprs': {},
           'expected_compatible': True,
           'mode': 'result'},
 'ex029': {'expression': 'torch.where(mask, x, y)',
           'intermediates': {},
           'audit': ['mask', 'x', 'y'],
           'value_exprs': {},
           'expected_compatible': True,
           'mode': 'basic'},
 'ex030': {'expression': 'activations + bias',
           'intermediates': {},
           'audit': ['activations', 'bias'],
           'value_exprs': {},
           'expected_compatible': True,
           'mode': 'basic'},
 'ex031': {'expression': 'offset_view + activations',
           'intermediates': {'offset_view': 'offset[:, None]'},
           'audit': ['activations', 'offset'],
           'value_exprs': {},
           'expected_compatible': True,
           'mode': 'result'},
 'ex032': {'expression': 'tokens + positions',
           'intermediates': {},
           'audit': ['tokens', 'positions'],
           'value_exprs': {},
           'expected_compatible': True,
           'mode': 'basic'},
 'ex033': {'expression': 'mask_view * embeddings',
           'intermediates': {'mask_view': 'mask[:, :, None]'},
           'audit': ['mask', 'embeddings'],
           'value_exprs': {},
           'expected_compatible': True,
           'mode': 'result'},
 'ex034': {'expression': 'losses * class_weights',
           'intermediates': {},
           'audit': ['losses', 'class_weights'],
           'value_exprs': {},
           'expected_compatible': True,
           'mode': 'basic'},
 'ex035': {'expression': 'bias_view + images',
           'intermediates': {'bias_view': 'bias[None, :, None, None]'},
           'audit': ['images', 'bias'],
           'value_exprs': {},
           'expected_compatible': True,
           'mode': 'result'},
 'ex036': {'expression': 'scores.masked_fill(~mask, -1e9)',
           'intermediates': {'inverted_mask': '~mask'},
           'audit': ['scores', 'mask'],
           'value_exprs': {},
           'expected_compatible': True,
           'mode': 'basic'},
 'ex037': {'expression': 'x_rows - y_rows',
           'intermediates': {'x_rows': 'x[:, None, :]', 'y_rows': 'y[None, :, :]'},
           'audit': ['x', 'y'],
           'value_exprs': {},
           'expected_compatible': True,
           'mode': 'result'},
 'ex038': {'expression': 'source.expand_as(target)',
           'intermediates': {},
           'audit': ['source', 'target'],
           'value_exprs': {},
           'expected_compatible': True,
           'mode': 'result'},
 'ex039': {'expression': 'torch.broadcast_to(source, target.shape)',
           'intermediates': {},
           'audit': ['source', 'target'],
           'value_exprs': {},
           'expected_compatible': True,
           'mode': 'result'},
 'ex040': {'expression': 'source.repeat(2, 1)',
           'intermediates': {},
           'audit': ['source'],
           'value_exprs': {},
           'expected_compatible': True,
           'mode': 'result'},
 'ex041': {'expression': 'source.expand(2, 3)',
           'intermediates': {},
           'audit': ['source'],
           'value_exprs': {'shares_storage': 'out.untyped_storage().data_ptr() == '
                                             'source.untyped_storage().data_ptr()'},
           'expected_compatible': True,
           'mode': 'storage'},
 'ex042': {'expression': 'source.expand(2, 3).clone()',
           'intermediates': {},
           'audit': ['source'],
           'value_exprs': {'shares_storage': 'out.untyped_storage().data_ptr() == '
                                             'source.untyped_storage().data_ptr()'},
           'expected_compatible': True,
           'mode': 'storage'},
 'ex043': {'expression': 'normalized',
           'intermediates': {'mean': 'x.mean(dim=0, keepdim=True)',
                             'centered': 'x - mean',
                             'variance': '(centered ** 2).mean(dim=0, keepdim=True)',
                             'inv_std': '(variance + 1e-5).rsqrt()',
                             'normalized': 'centered * inv_std'},
           'audit': ['x'],
           'value_exprs': {},
           'expected_compatible': True,
           'mode': 'capstone'},
 'ex044': {'expression': 'masked_scores',
           'intermediates': {'mask_view': 'causal[None, None, :, :]',
                             'masked_scores': 'scores.masked_fill(~mask_view, -1e9)'},
           'audit': ['scores', 'causal'],
           'value_exprs': {},
           'expected_compatible': True,
           'mode': 'capstone'},
 'ex045': {'expression': 'sq_distance',
           'intermediates': {'x_rows': 'x[:, None, :]',
                             'y_rows': 'y[None, :, :]',
                             'difference': 'x_rows - y_rows',
                             'sq_distance': '(difference ** 2).sum(dim=2)'},
           'audit': ['x', 'y'],
           'value_exprs': {},
           'expected_compatible': True,
           'mode': 'capstone'}}

PRE_OPERATOR_GROUPS = {'ex009': [('a_before_op', 'a'), ('b_before_op', 'b')],
 'ex010': [('a_before_op', 'a'), ('b_before_op', 'b')],
 'ex011': [('a_before_op', 'a'), ('b_before_op', 'b')],
 'ex012': [('a_before_op', 'a'), ('b_before_op', 'b')],
 'ex013': [('a_before_op', 'a'), ('b_before_op', 'b')],
 'ex014': [('a_before_op', 'a'), ('b_before_op', 'b')],
 'ex015': [('a_before_op', 'a'), ('b_before_op', 'b')],
 'ex016': [('a_before_op', 'a'), ('b_before_op', 'b')],
 'ex017': [('matrix_before_op', 'matrix'), ('row_before_op', 'row')],
 'ex018': [('matrix_before_op', 'matrix'), ('column_before_op', 'column')],
 'ex019': [('column_before_op', 'column'), ('row_before_op', 'row')],
 'ex020': [('rows_col_before_op', 'rows_col'), ('grid_before_op', 'grid')],
 'ex021': [('cols_row_before_op', 'cols_row'), ('grid_before_op', 'grid')],
 'ex022': [('prepared_before_op', 'prepared'), ('grid_before_op', 'grid')],
 'ex023': [('prepared_before_op', 'prepared'), ('grid_before_op', 'grid')],
 'ex024': [('channel_view_before_op', 'channel_view'), ('images_before_op', 'images')],
 'ex025': [('matrix_before_op', 'matrix'), ('vector_before_op', 'vector')],
 'ex026': [('images_before_op', 'images'), ('background_before_op', 'background')],
 'ex027': [('a_before_op', 'a'), ('b_before_op', 'b')],
 'ex028': [('prepared_before_op', 'prepared'), ('images_before_op', 'images')],
 'ex029': [('mask_before_op', 'mask'), ('x_before_op', 'x'), ('y_before_op', 'y')],
 'ex033': [('mask_view_before_op', 'mask_view'), ('embeddings_before_op', 'embeddings')],
 'ex036': [('scores_before_op', 'scores'), ('inverted_mask_before_op', 'inverted_mask')],
 'ex037': [('x_rows_before_op', 'x_rows'), ('y_rows_before_op', 'y_rows')]}


def _clone(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    return value


def _aligned_shape(shape, rank):
    return (1,) * (rank - len(shape)) + tuple(shape)


def _incompatible_axes(shapes):
    rank = max((len(shape) for shape in shapes), default=0)
    aligned = [_aligned_shape(shape, rank) for shape in shapes]
    bad = []
    for axis in range(rank):
        non_singletons = {shape[axis] for shape in aligned if shape[axis] != 1}
        if len(non_singletons) > 1:
            bad.append(axis)
    return tuple(bad)


def build_reference(key, supplied):
    case = CASES[key]
    scope = dict(supplied)
    reference = {}

    for name, value in supplied.items():
        if isinstance(value, torch.Tensor):
            reference[f"{name}_shape"] = tuple(value.shape)

    for name, expression in case["intermediates"].items():
        value = eval(expression, {"torch": torch}, scope)
        scope[name] = value
        reference[name] = _clone(value)
        if isinstance(value, torch.Tensor):
            reference[f"{name}_shape"] = tuple(value.shape)

    audit_names = case["audit"]
    audit_tensors = [scope[name] for name in audit_names]
    audit_shapes = [tuple(tensor.shape) for tensor in audit_tensors]

    output = None
    error = None
    try:
        output = eval(case["expression"], {"torch": torch}, scope)
    except RuntimeError as exc:
        error = exc

    compatible = error is None
    assert compatible is case["expected_compatible"], (key, error)
    reference["compatible"] = compatible

    if not compatible:
        rank = max((len(shape) for shape in audit_shapes), default=0)
        for name, shape in zip(audit_names, audit_shapes, strict=True):
            reference[f"{name}_aligned_shape"] = _aligned_shape(shape, rank)
        reference["incompatible_axes"] = _incompatible_axes(audit_shapes)
        return reference

    assert isinstance(output, torch.Tensor), key
    scope["out"] = output
    reference["out"] = _clone(output)
    reference["out_shape"] = tuple(output.shape)

    if key in PRE_OPERATOR_GROUPS:
        group = PRE_OPERATOR_GROUPS[key]
        operator_values = [scope[scope_name] for _, scope_name in group]
        operator_ready = torch.broadcast_tensors(*operator_values)
        for (field, _), value in zip(group, operator_ready, strict=True):
            reference[field] = _clone(value)

    if case["mode"] == "aligned":
        rank = output.ndim
        for name, tensor in zip(audit_names, audit_tensors, strict=True):
            reference[f"{name}_aligned_shape"] = _aligned_shape(tensor.shape, rank)

    for field, expression in case["value_exprs"].items():
        reference[field] = eval(expression, {"torch": torch}, scope)

    return reference
