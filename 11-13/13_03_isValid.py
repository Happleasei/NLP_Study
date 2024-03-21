def isValid(s):
	stack = []
	dic = {'[':']', '{':'}', '(':')'}
	for c in s:
		if c in dic.values():
			stack.append(c)
		elif c in dic.keys():
			if stack==[] or dic[c]!=stack.pop():
				return False
		else:
			return False
	return stack==[]
