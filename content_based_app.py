import streamlit as st
import pandas as pd
import re
import pickle

# Load your datasets
san_pham = pd.read_csv('C:/Users/PC/Documents/NamNC/GUI_Recommender_System/San_pham.csv')
khach_hang = pd.read_csv('C:/Users/PC/Documents/NamNC/GUI_Recommender_System/Khach_hang.csv')
danh_gia = pd.read_csv('C:/Users/PC/Documents/NamNC/GUI_Recommender_System/Danh_gia.csv')

# # Load stopwords
# with open('vietnamese-stopwords.txt', 'r', encoding="utf8") as file:
#     stopwords = file.read().split('\n')


# Data preprocessing and merging
join_san_pham_danh_gia = pd.merge(danh_gia, san_pham, on='ma_san_pham', how='left')
# join_san_pham_danh_gia['ma_khach_hang'] = join_san_pham_danh_gia['ma_khach_hang'].astype(str)
df_ori = pd.merge(join_san_pham_danh_gia, khach_hang, on='ma_khach_hang', how='left')
dataframe = df_ori.copy()

df = dataframe[['ma_san_pham', 'ten_san_pham', 'mo_ta', 'so_sao']]

# function cần thiết

# Load cosine similarity matrix from the pickle file
with open('C:/Users/PC/Documents/NamNC/GUI_Recommender_System/products_cosine_sim.pkl', 'rb') as f:
    cosine_sim_new = pickle.load(f)

# Recommendation function
def get_recommendations_cosine(sp_id_or_name, cosine_sim=cosine_sim_new, nums=10, min_rating=4):
    if isinstance(sp_id_or_name, int):
        if sp_id_or_name not in df['ma_san_pham'].values:
            return pd.DataFrame()
        idx = df.index[df['ma_san_pham'] == sp_id_or_name][0]
    else:
        matching_products = df[df['ten_san_pham'].str.contains(sp_id_or_name, case=False, na=False) |
                               df['mo_ta'].str.contains(sp_id_or_name, case=False, na=False)]
        if matching_products.empty:
            return pd.DataFrame()
        idx = matching_products.index[0]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:nums + 1]

    sp_indices = [i[0] for i in sim_scores]
    recommended_products = df[['ma_san_pham', 'ten_san_pham', 'mo_ta', 'so_sao']].iloc[sp_indices]
    recommended_products = recommended_products[recommended_products['so_sao'] >= min_rating]
    recommended_products = recommended_products.drop_duplicates(subset='ma_san_pham').sort_values(by='so_sao', ascending=False)

    return recommended_products
   
st.title("Data Science Project")
st.write("##")

menu = ["Yêu cầu bài toán", "Xây dựng model", "Gợi ý cho người dùng"]
choice = st.sidebar.selectbox('Menu', menu)
st.sidebar.write("""#### Thành viên thực hiện:
                 1) CHẾ THỊ ÁNH TUYỀN
                 2) NGUYỄN CHẤN NAM""")
st.sidebar.write("""#### Giảng viên hướng dẫn: """)
st.sidebar.write("""#### Ngày báo cáo đồ án: 12/2024""")
if choice == 'Yêu cầu bài toán':

    st.subheader("Yêu cầu bài toán")
    st.write("""
    ### HASAKI.VN là hệ thống cửa hàng mỹ phẩm chính hãng và dịch vụ chăm sóc sắc đẹp chuyên sâu với hệ thống cửa hàng trải dài trên toàn quốc; và hiện đang là đối tác phân phối chiến lược tại thị trường Việt Nam của hàng loạt thương hiệu lớn./n
    ### Khách hàng có thể lên đây để lựa chọn sản phẩm, xem các đánh giá/ nhận xét cũng như đặt mua sản phẩm.
    ### HASAKI.VN chưa triển khai hệ thống Recommender System giúp đề xuất sản phẩm phù hợp tới người dùng.
    """)  
    st.write("""## => Vấn đề/ Yêu cầu: Sử dụng thuật toán Machine Learning trong python để gợi ý đề xuất sản phẩm phù hợp với người dùng dựa trên content based filtering.""")

elif choice == 'Xây dựng model':
    st.subheader("Xây dựng model")
    st.write("##### 1. Dữ liệu mẫu")
    san_pham['ma_san_pham'] = san_pham['ma_san_pham'].astype(str)
    st.dataframe(san_pham.head(5))
    st.dataframe(san_pham.tail(5))
    # st.write("##### 2. Visualize Ham and Spam")
    # fig1 = sns.countplot(data=data[['v1']], x='v1')    
    # st.pyplot(fig1.figure)

    st.write("##### 3. Xây dựng model...")
    st.write("##### 4. Evaluation")
    # st.code("Score train:"+ str(round(score_train,2)) + " vs Score test:" + str(round(score_test,2)))
    # st.code("Accuracy:"+str(round(acc,2)))




    st.write("##### 5. Tổng kết: mô hình này chấp nhận được cho gợi ý đề xuất sản phẩm phù hợp với người dùng.")

elif choice == 'Gợi ý cho người dùng':

    # Streamlit UI
    st.image('C:/Users/PC/Documents/NamNC/GUI_Recommender_System/hasaki_banner.jpg')
    st.title('Hệ thống gợi ý sản phẩm')

    # Display first 10 products
    st.subheader('Danh sách sản phẩm')
    san_pham['ma_san_pham'] = san_pham['ma_san_pham'].astype(str)
    st.dataframe(san_pham.head(10))  # Display the first 10 products
    

    user_input = st.text_input("Nhập tên sản phẩm, mã sản phẩm hoặc nội dung mô tả sản phẩm:")

    # Check if the user input is empty or only whitespace
    is_input_empty = user_input.strip() == ""

    # Display the "Gợi ý" button, disable it if input is empty
    if st.button('Gợi ý', disabled=is_input_empty):
        try:
            # Kiểm tra xem có phải là mã sản phẩm hay không
            user_input_int = int(user_input)
            recommendations = get_recommendations_cosine(user_input_int)

            # Tìm thông tin sản phẩm tương ứng
            product_info = df[df['ma_san_pham'] == user_input_int]
            if not product_info.empty:
                product_name = product_info['ten_san_pham'].values[0]
                product_desc = product_info['mo_ta'].values[0]
                product_rating = product_info['so_sao'].values[0]
                
                # Hiển thị thông tin sản phẩm
                st.write(f"**I. Tên sản phẩm tương ứng mã sản phẩm vừa nhập:** {product_name}")
                st.write(f"**II. Mô tả:** {product_desc}")
                st.write(f"**III. Điểm trung bình:** {product_rating}")

                # Thống kê số lượng đánh giá theo từng sao
                star_ratings_count = danh_gia[danh_gia['ma_san_pham'] == user_input_int]['so_sao'].value_counts().sort_index()
                st.write("**IV. Số lượng đánh giá theo từng sao:**")
                for star in range(1, 6):
                    count = star_ratings_count.get(star, 0)
                    st.write(f"{star} sao: {count} đánh giá")

        except ValueError:
            # Nếu không, xem như nhập vào là tên sản phẩm
            recommendations = get_recommendations_cosine(user_input)
                        
        # Hiển thị kết quả
        if not recommendations.empty:
            st.write("**V. Top 10 sản phẩm gợi ý:**")
            
            # Chuyển đổi cột 'ma_san_pham' sang kiểu chuỗi
            recommendations['ma_san_pham'] = recommendations['ma_san_pham'].astype(str)
            
            # Hiển thị DataFrame
            st.dataframe(recommendations[['ma_san_pham', 'ten_san_pham', 'mo_ta', 'so_sao']])
        else:
            st.write("Không tìm thấy sản phẩm gợi ý thỏa điều kiện.")

    

