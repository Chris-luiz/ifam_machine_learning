from django import template

register = template.Library()

@register.filter
def get_item(dictionary, key):
    """Obtém um item de um dicionário"""
    if isinstance(dictionary, dict):
        return dictionary.get(key, '')
    return ''

@register.filter
def mul(value, arg):
    """Multiplica um valor por outro"""
    try:
        return float(value) * float(arg)
    except (ValueError, TypeError):
        return 0