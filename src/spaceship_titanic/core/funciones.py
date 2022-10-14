from pandas import DataFrame


def validate_classification(columns_classification, data: DataFrame):
    item_list = []
    for _, items in columns_classification.items():
        item_list += items
    if len(item_list) != len(set(item_list)):
        raise Exception('Hay columnas duplicadas en columns classification.')
    item_set = set(item_list)
    columns_set = set(data.columns)
    columns_not_clasiffied = columns_set.difference(item_set)
    innexistent_classified = item_set.difference(columns_set)
    if len(columns_not_clasiffied) > 0:
        raise Exception(f'Las siguientes columnas no están clasificadas {columns_not_clasiffied}')
    if len(innexistent_classified) > 0:
        raise Exception(f'Las siguientes columnas clasificadas no existen en el dataset {innexistent_classified}')
