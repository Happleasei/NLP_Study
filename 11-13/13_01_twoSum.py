def twoSum(nums, target):
	dic = {}
	for i in range(len(nums)):
		if nums[i] not in dic:
			dic[target-nums[i]] = i
		else:
			return dic[nums[i]], i
	return -1, -1
