class ModelTools:
    
	# Trie un dictionnaire par ses valeurs (ordre croissant).
	# lambda est une fonction anonyme : lambda item: item[1] extrait la valeur
	# de chaque paire (clé, valeur) pour l'utiliser comme critère de tri.
	@staticmethod
	def sorted_dict_by_values(dict_object):
		sorted_dict = dict(sorted(dict_object.items(), key=lambda item: item[1]))
		return sorted_dict