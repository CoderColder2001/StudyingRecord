## 链表
### 题目
---
### &emsp; 143. 重排链表 MID
关键思路：  
- 时间O（N）、空间O（1）的解法
- 先用<b>快慢指针定位中点</b>
- 翻转后半部分链表，再将链表前后部分合并

<details> 
<summary> <b>C++ Code</b> </summary>

```c++
class Solution {
public:
    void reorderList(ListNode* head) {
        if(head == nullptr)
            return;
        ListNode* mid = middleNode(head);
        ListNode* l1 = head;
        ListNode* l2 = mid->next;
        mid->next = nullptr;
        l2 = reverseList(l2);
        mergeList(l1, l2);
    }

    ListNode* middleNode(ListNode* head) // 快慢指针找中点
    {
        ListNode* slow = head;
        ListNode* fast = head;
        while(fast->next != nullptr && fast->next->next != nullptr)
        {
            slow = slow->next;
            fast = fast->next->next;
        }
        return slow;
    }
    ListNode* reverseList(ListNode* head)
    {
        ListNode* prev = nullptr;
        ListNode* cur = head;
        while(cur != nullptr)
        {
            ListNode* curNext = cur->next;
            cur->next = prev;
            prev = cur;
            cur = curNext;
        }
        return prev; // 新的head
    }
    void mergeList(ListNode* l1, ListNode* l2)
    {
        ListNode* l1Next, * l2Next;
        while(l1 != nullptr && l2 != nullptr)
        {
            l1Next = l1->next;
            l2Next = l2->next;
            l1->next = l2;
            l1 = l1Next;
            l2->next = l1;
            l2 = l2Next;
        }
    }
};
```
</details> 
<br>
