#include "chainerx/stack_vector.h"

#include <algorithm>
#include <cstdint>
#include <list>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace chainerx {
namespace {

static_assert(
        std::is_same<std::iterator_traits<StackVector<int, 5>::iterator>::iterator_category, std::random_access_iterator_tag>::value, "");

TEST(StackVectorTest, Operations) {
    using Vector = StackVector<int, 5>;
    // ctor (default)
    Vector vec{};

    // max_size
    EXPECT_EQ(size_t{5}, vec.max_size());
    // size
    EXPECT_EQ(size_t{0}, vec.size());
    // empty
    EXPECT_TRUE(vec.empty());

    // emplace / emplace_back
    vec.emplace(vec.cbegin(), 4);
    vec.emplace_back(-1);
    vec.emplace_back(5);
    vec.emplace(vec.cbegin() + 1, 3);
    // size
    EXPECT_EQ(size_t{4}, vec.size());
    // empty
    EXPECT_FALSE(vec.empty());
    // operator[] const
    EXPECT_EQ(4, vec[0]);
    EXPECT_EQ(3, vec[1]);
    EXPECT_EQ(-1, vec[2]);
    EXPECT_EQ(5, vec[3]);

    // ctor (initializer_list)
    Vector vec2{4, 3, -1, 5};
    // operator==
    EXPECT_EQ(vec2, vec);
    // operator[] non-const
    vec2[1] = 2;
    // operator!=
    EXPECT_NE(vec2, vec);

    {
        const Vector& cvec = vec;

        // begin, end
        EXPECT_EQ(std::vector<int>({4, 3, -1, 5}), std::vector<int>({vec.begin(), vec.end()}));
        EXPECT_EQ(std::vector<int>({4, 3, -1, 5}), std::vector<int>({cvec.begin(), cvec.end()}));

        // cbegin, cend
        EXPECT_EQ(std::vector<int>({4, 3, -1, 5}), std::vector<int>({vec.cbegin(), vec.cend()}));
        EXPECT_EQ(std::vector<int>({4, 3, -1, 5}), std::vector<int>({cvec.cbegin(), cvec.cend()}));

        // rbegin, rend
        EXPECT_EQ(std::vector<int>({5, -1, 3, 4}), std::vector<int>({vec.rbegin(), vec.rend()}));
        EXPECT_EQ(std::vector<int>({5, -1, 3, 4}), std::vector<int>({cvec.rbegin(), cvec.rend()}));

        // crbegin, crend
        EXPECT_EQ(std::vector<int>({5, -1, 3, 4}), std::vector<int>({vec.crbegin(), vec.crend()}));
        EXPECT_EQ(std::vector<int>({5, -1, 3, 4}), std::vector<int>({cvec.crbegin(), cvec.crend()}));
    }

    // sort
    std::sort(vec.begin(), vec.end());
    EXPECT_EQ(Vector({-1, 3, 4, 5}), vec);

    // erase
    vec.erase(vec.begin() + 1, vec.begin() + 3);
    EXPECT_EQ(Vector({-1, 5}), vec);

    // insert
    {
        int data[] = {6, 2};
        vec.insert(vec.begin() + 1, data, data + 2);
        EXPECT_EQ(Vector({-1, 6, 2, 5}), vec);
    }

    // clear
    vec.clear();
    EXPECT_EQ(size_t{5}, vec.max_size());
    EXPECT_EQ(size_t{0}, vec.size());
    EXPECT_EQ(Vector{}, vec);
    EXPECT_TRUE(vec.empty());
}

TEST(StackVectorTest, DefaultCtor) {
    StackVector<int, 5> vec1{};
    EXPECT_EQ(size_t{5}, vec1.max_size());
    EXPECT_EQ(size_t{0}, vec1.size());
    EXPECT_TRUE(vec1.empty());

    StackVector<int, 0> vec2{};
    EXPECT_EQ(size_t{0}, vec2.max_size());
    EXPECT_EQ(size_t{0}, vec2.size());
    EXPECT_TRUE(vec2.empty());
}

TEST(StackVectorTest, InitializerListCtor) {
    StackVector<int, 5> vec1{2, 3};
    EXPECT_EQ(size_t{5}, vec1.max_size());
    EXPECT_EQ(size_t{2}, vec1.size());
    EXPECT_EQ(2, vec1[0]);
    EXPECT_EQ(3, vec1[1]);
    EXPECT_FALSE(vec1.empty());

    StackVector<int, 5> vec2{2, 3, 6, 1, 9};
    EXPECT_EQ(size_t{5}, vec2.max_size());
    EXPECT_EQ(size_t{5}, vec2.size());
    EXPECT_EQ(2, vec2[0]);
    EXPECT_EQ(3, vec2[1]);
    EXPECT_EQ(6, vec2[2]);
    EXPECT_EQ(1, vec2[3]);
    EXPECT_EQ(9, vec2[4]);
    EXPECT_FALSE(vec2.empty());
}

TEST(StackVectorTest, RangeCtor) {
    std::list<int> data{2, 3};  // a container whose iterators are not simple pointers

    StackVector<int, 5> vec1{data.begin(), data.end()};
    EXPECT_EQ(size_t{5}, vec1.max_size());
    EXPECT_EQ(size_t{2}, vec1.size());
    EXPECT_EQ(2, vec1[0]);
    EXPECT_EQ(3, vec1[1]);
    EXPECT_FALSE(vec1.empty());
}

TEST(StackVectorTest, CopyCtor) {
    StackVector<int, 5> vec1{2, 3};
    StackVector<int, 5> vec2 = vec1;
    EXPECT_EQ(size_t{5}, vec2.max_size());
    EXPECT_EQ(size_t{2}, vec2.size());
    EXPECT_EQ(2, vec2[0]);
    EXPECT_EQ(3, vec2[1]);
    EXPECT_FALSE(vec2.empty());
}

TEST(StackVectorTest, CopyAssign) {
    StackVector<int, 5> vec1{2, 3};
    StackVector<int, 5> vec2{};
    vec2 = vec1;
    EXPECT_EQ(size_t{5}, vec1.max_size());
    EXPECT_EQ(size_t{2}, vec1.size());
    EXPECT_EQ(2, vec1[0]);
    EXPECT_EQ(3, vec1[1]);
    EXPECT_FALSE(vec1.empty());
}

TEST(StackVectorTest, MoveCtor) {
    StackVector<int, 5> vec1{2, 3};
    StackVector<int, 5> vec2 = std::move(vec1);
    EXPECT_EQ(size_t{5}, vec2.max_size());
    EXPECT_EQ(size_t{2}, vec2.size());
    EXPECT_EQ(2, vec2[0]);
    EXPECT_EQ(3, vec2[1]);
    EXPECT_FALSE(vec2.empty());
}

TEST(StackVectorTest, MoveAssign) {
    StackVector<int, 5> vec1{2, 3};
    StackVector<int, 5> vec2{};
    vec2 = std::move(vec1);
    EXPECT_EQ(size_t{5}, vec2.max_size());
    EXPECT_EQ(size_t{2}, vec2.size());
    EXPECT_EQ(2, vec2[0]);
    EXPECT_EQ(3, vec2[1]);
    EXPECT_FALSE(vec2.empty());
}

TEST(StackVectorTest, Equality) {
    StackVector<int, 5> vec1{2, 3};
    StackVector<int, 5> vec2{2, 3};
    EXPECT_EQ(vec1, vec2);
}

TEST(StackVectorTest, EqualityEmpty) {
    StackVector<int, 5> vec1{};
    StackVector<int, 5> vec2{};
    EXPECT_EQ(vec1, vec2);
}

TEST(StackVectorTest, Inequality1) {
    StackVector<int, 5> vec1{2, 3};
    StackVector<int, 5> vec2{2, 3, 4};
    EXPECT_NE(vec1, vec2);
}

TEST(StackVectorTest, Inequality2) {
    StackVector<int, 5> vec1{2, 3};
    StackVector<int, 5> vec2{};
    EXPECT_NE(vec1, vec2);
}

TEST(StackVectorTest, PushBack) {
    StackVector<int, 5> vec1{};
    vec1.push_back(2);
    vec1.push_back(3);

    EXPECT_EQ(size_t{2}, vec1.size());
    EXPECT_EQ(2, vec1[0]);
    EXPECT_EQ(3, vec1[1]);
    EXPECT_FALSE(vec1.empty());
}

TEST(StackVectorTest, EmplaceBack) {
    // Note: how the elements are constructed is tested in the "CtorCallCounts" test below.
    StackVector<int, 5> vec1{};
    vec1.emplace_back(2);
    vec1.emplace_back(3);

    EXPECT_EQ(size_t{2}, vec1.size());
    EXPECT_EQ(2, vec1[0]);
    EXPECT_EQ(3, vec1[1]);
    EXPECT_FALSE(vec1.empty());
}

TEST(StackVectorTest, Emplace) {
    StackVector<int, 5> vec1{2, 3};

    vec1.emplace(vec1.end(), 5);  // emplace at end
    EXPECT_EQ(size_t{3}, vec1.size());
    EXPECT_EQ(2, vec1[0]);
    EXPECT_EQ(3, vec1[1]);
    EXPECT_EQ(5, vec1[2]);

    vec1.emplace(vec1.begin() + 1, 9);  // emplace at middle
    EXPECT_EQ(size_t{4}, vec1.size());
    EXPECT_EQ(2, vec1[0]);
    EXPECT_EQ(9, vec1[1]);
    EXPECT_EQ(3, vec1[2]);
    EXPECT_EQ(5, vec1[3]);

    vec1.emplace(vec1.begin(), 7);  // emplace at begin
    EXPECT_EQ(size_t{5}, vec1.size());
    EXPECT_EQ(7, vec1[0]);
    EXPECT_EQ(2, vec1[1]);
    EXPECT_EQ(9, vec1[2]);
    EXPECT_EQ(3, vec1[3]);
    EXPECT_EQ(5, vec1[4]);
}

TEST(StackVectorTest, Clear) {
    StackVector<int, 5> vec1{2, 3};
    vec1.clear();

    EXPECT_EQ(size_t{0}, vec1.size());
    EXPECT_TRUE(vec1.empty());
}

TEST(StackVectorTest, Resize) {
    StackVector<int, 5> vec1{2, 3};

    // shrink
    vec1.resize(1);
    EXPECT_EQ(size_t{1}, vec1.size());
    EXPECT_EQ(2, vec1[0]);
    EXPECT_FALSE(vec1.empty());

    // expand
    vec1.resize(3);
    EXPECT_EQ(size_t{3}, vec1.size());
    EXPECT_EQ(2, vec1[0]);
    EXPECT_EQ(0, vec1[1]);
    EXPECT_EQ(0, vec1[2]);
    EXPECT_FALSE(vec1.empty());

    // shrink to zero
    vec1.resize(0);
    EXPECT_EQ(size_t{0}, vec1.size());
    EXPECT_TRUE(vec1.empty());
}

TEST(StackVectorTest, Data) {
    {
        StackVector<int, 5> vec{2, 3};
        const StackVector<int, 5>& cvec = vec;
        EXPECT_EQ(&vec[0], vec.data());
        EXPECT_EQ(&vec[0], cvec.data());
    }

    // For empty vector, return values can be arbitrary but should not throw
    {
        StackVector<int, 5> vec{};
        const StackVector<int, 5>& cvec = vec;
        vec.data();  // no throw
        cvec.data();
    }
}

TEST(StackVectorTest, Front) {
    StackVector<int, 5> vec1{2, 3};

    EXPECT_EQ(2, vec1.front());
    vec1.front() = 9;

    EXPECT_EQ(size_t{2}, vec1.size());
    EXPECT_EQ(9, vec1.front());
    EXPECT_EQ(9, vec1[0]);
    EXPECT_EQ(3, vec1[1]);
}

TEST(StackVectorTest, Back) {
    StackVector<int, 5> vec1{2, 3};

    EXPECT_EQ(3, vec1.back());
    vec1.back() = 9;

    EXPECT_EQ(size_t{2}, vec1.size());
    EXPECT_EQ(9, vec1.back());
    EXPECT_EQ(2, vec1[0]);
    EXPECT_EQ(9, vec1[1]);
}

TEST(StackVectorTest, Iterator) {
    StackVector<int, 5> vec1{2, 3};

    EXPECT_EQ(vec1.begin() + 2, vec1.end());
    EXPECT_EQ(vec1.end() - 2, vec1.begin());

    auto it = vec1.begin();
    EXPECT_EQ(2, *it);

    *it = 9;
    EXPECT_EQ(9, *it);
    EXPECT_EQ(9, vec1[0]);
    EXPECT_EQ(3, vec1[1]);

    {
        auto it2 = ++it;
        EXPECT_EQ(it, it2);
        EXPECT_EQ(3, *it);
        EXPECT_EQ(vec1.begin() + 1, it);
    }

    {
        auto it2 = --it;
        EXPECT_EQ(it, it2);
        EXPECT_EQ(9, *it);
        EXPECT_EQ(vec1.begin(), it);
    }

    {
        auto it2 = it++;
        EXPECT_EQ(it - 1, it2);
        EXPECT_EQ(3, *it);
        EXPECT_EQ(vec1.begin() + 1, it);
    }

    {
        auto it2 = it--;
        EXPECT_EQ(it + 1, it2);
        EXPECT_EQ(9, *it);
        EXPECT_EQ(vec1.begin(), it);
    }
}

TEST(StackVectorTest, EraseMiddle) {
    using Vector = StackVector<int, 5>;
    Vector vec1{2, 3, 4, 5};

    // Erase 2 elements in the middle
    auto it = vec1.erase(vec1.begin() + 1, vec1.begin() + 3);
    EXPECT_EQ(size_t{2}, vec1.size());
    EXPECT_EQ(Vector({2, 5}), vec1);
    EXPECT_EQ(vec1.begin() + 1, it);
}

TEST(StackVectorTest, EraseEnd) {
    using Vector = StackVector<int, 5>;
    Vector vec1{2, 3, 4, 5};

    // Erase 2 elements at end
    auto it = vec1.erase(vec1.begin() + 2, vec1.end());
    EXPECT_EQ(size_t{2}, vec1.size());
    EXPECT_EQ(Vector({2, 3}), vec1);
    EXPECT_EQ(vec1.end(), it);
}

TEST(StackVectorTest, EraseBegin) {
    using Vector = StackVector<int, 5>;
    Vector vec1{2, 3, 4, 5};

    // Erase 2 elements at begin
    auto it = vec1.erase(vec1.begin(), vec1.begin() + 2);
    EXPECT_EQ(size_t{2}, vec1.size());
    EXPECT_EQ(Vector({4, 5}), vec1);
    EXPECT_EQ(vec1.begin(), it);
}

TEST(StackVectorTest, EraseEmpty) {
    using Vector = StackVector<int, 5>;
    Vector vec1{2, 3, 4, 5};

    // Erase empty range
    auto it = vec1.erase(vec1.begin() + 2, vec1.begin() + 2);
    EXPECT_EQ(size_t{4}, vec1.size());
    EXPECT_EQ(Vector({2, 3, 4, 5}), vec1);
    EXPECT_EQ(vec1.begin() + 2, it);
}

TEST(StackVectorTest, CtorCallCounts) {
    static int n_default_ctor{};
    static int n_copy_ctor{};
    static int n_move_ctor{};
    static int n_normal_ctor{};
    static int n_copy_assign{};
    static int n_move_assign{};

    struct Element {
        Element() { ++n_default_ctor; }
        Element(const Element& /*other*/) { ++n_copy_ctor; }
        Element(Element&& /*other*/) noexcept { ++n_move_ctor; }
        Element(const char* /*param1*/, const char* /*param2*/) { ++n_normal_ctor; }
        Element& operator=(const Element& /*other*/) {
            ++n_copy_assign;
            return *this;
        }
        Element& operator=(Element&& /*other*/) noexcept {
            ++n_move_assign;
            return *this;
        }
    };

    auto reset = [&]() {
        n_default_ctor = 0;
        n_copy_ctor = 0;
        n_move_ctor = 0;
        n_normal_ctor = 0;
        n_copy_assign = 0;
        n_move_assign = 0;
    };

    auto create_vec = [&]() {
        StackVector<Element, 5> vec{};
        vec.emplace_back("a", "b");
        vec.emplace_back("c", "d");
        return vec;
    };

    // default ctor
    {
        reset();
        StackVector<Element, 5> vec{};
        (void)vec;  // unused
    }
    EXPECT_EQ(5, n_default_ctor);
    EXPECT_EQ(0, n_copy_ctor);
    EXPECT_EQ(0, n_move_ctor);
    EXPECT_EQ(0, n_normal_ctor);
    EXPECT_EQ(0, n_copy_assign);
    EXPECT_EQ(0, n_move_assign);

    // emplace_back
    {
        StackVector<Element, 5> vec{};
        reset();
        vec.emplace_back("a", "b");
        vec.emplace_back("c", "d");
    }
    EXPECT_EQ(0, n_default_ctor);
    EXPECT_EQ(0, n_copy_ctor);
    EXPECT_EQ(0, n_move_ctor);
    EXPECT_EQ(2, n_normal_ctor);
    EXPECT_EQ(0, n_copy_assign);
    EXPECT_EQ(2, n_move_assign);

    // copy ctor
    {
        StackVector<Element, 5> vec = create_vec();
        reset();
        StackVector<Element, 5> vec2 = vec;
        (void)vec2;  // unused
    }
    EXPECT_EQ(5, n_default_ctor);
    EXPECT_EQ(0, n_copy_ctor);
    EXPECT_EQ(0, n_move_ctor);
    EXPECT_EQ(0, n_normal_ctor);
    EXPECT_EQ(2, n_copy_assign);
    EXPECT_EQ(0, n_move_assign);

    // copy assignment
    {
        StackVector<Element, 5> vec = create_vec();
        StackVector<Element, 5> vec2;
        reset();
        vec2 = vec;
    }
    EXPECT_EQ(0, n_default_ctor);
    EXPECT_EQ(0, n_copy_ctor);
    EXPECT_EQ(0, n_move_ctor);
    EXPECT_EQ(0, n_normal_ctor);
    EXPECT_EQ(2, n_copy_assign);
    EXPECT_EQ(0, n_move_assign);

    // move ctor
    {
        StackVector<Element, 5> vec = create_vec();
        reset();
        StackVector<Element, 5> vec2 = std::move(vec);
        (void)vec2;  // unused
    }
    EXPECT_EQ(5, n_default_ctor);
    EXPECT_EQ(0, n_copy_ctor);
    EXPECT_EQ(0, n_move_ctor);
    EXPECT_EQ(0, n_normal_ctor);
    EXPECT_EQ(0, n_copy_assign);
    EXPECT_EQ(2, n_move_assign);

    // move assignment
    {
        StackVector<Element, 5> vec = create_vec();
        StackVector<Element, 5> vec2;
        reset();
        vec2 = std::move(vec);
    }
    EXPECT_EQ(0, n_default_ctor);
    EXPECT_EQ(0, n_copy_ctor);
    EXPECT_EQ(0, n_move_ctor);
    EXPECT_EQ(0, n_normal_ctor);
    EXPECT_EQ(0, n_copy_assign);
    EXPECT_EQ(2, n_move_assign);

    // resize (expand)
    {
        StackVector<Element, 5> vec = create_vec();
        reset();
        vec.resize(3);
    }
    EXPECT_EQ(1, n_default_ctor);
    EXPECT_EQ(0, n_copy_ctor);
    EXPECT_EQ(0, n_move_ctor);
    EXPECT_EQ(0, n_normal_ctor);
    EXPECT_EQ(0, n_copy_assign);
    EXPECT_EQ(1, n_move_assign);

    // resize (shrink)
    {
        StackVector<Element, 5> vec = create_vec();
        reset();
        vec.resize(1);
    }
    EXPECT_EQ(0, n_default_ctor);
    EXPECT_EQ(0, n_copy_ctor);
    EXPECT_EQ(0, n_move_ctor);
    EXPECT_EQ(0, n_normal_ctor);
    EXPECT_EQ(0, n_copy_assign);
    EXPECT_EQ(0, n_move_assign);

    // emplace
    {
        StackVector<Element, 5> vec = create_vec();
        reset();
        vec.emplace(vec.cend(), "x", "y");
    }
    EXPECT_EQ(0, n_default_ctor);
    EXPECT_EQ(0, n_copy_ctor);
    EXPECT_EQ(0, n_move_ctor);
    EXPECT_EQ(1, n_normal_ctor);
    EXPECT_EQ(0, n_copy_assign);
    EXPECT_EQ(1, n_move_assign);
}

}  // namespace
}  // namespace chainerx
