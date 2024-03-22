def isBalancedTree(root):
	def get_height(root):
		if not root:
			return 0
		left_height, right_height = get_height(root.left), get_height(root.right)
		if left_height < 0 or right_height < 0 or abs(left_height - right_height) > 1:
			return -1
		return max(left_height, right_height)+1
	return (get_height(root) >= 0)
