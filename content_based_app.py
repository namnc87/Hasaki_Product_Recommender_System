import streamlit as st
import pandas as pd
import re
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter

# CSS styles
st.markdown(
    """
    <style>
        body {
            background-color: #f5f5f5; /* Màu nền trang */
            color: #333333; /* Màu văn bản */
        }
        .big-font {
            font-size: 24px;
            font-weight: bold;
            color: #007bff; /* Màu chữ tên sản phẩm */
        }
        .desc-font {
            font-size: 16px;
            color: #555555; /* Màu mô tả sản phẩm */
        }
        .header {
            text-align: center;
            background-color: #28a745; /* Màu nền tiêu đề xanh lá cây */
            color: white;
            padding: 10px;
            border-radius: 5px;
            font-size: 44px; /* Thay đổi kích thước phông chữ */
        }
        .tab {
            background-color: white;
            border-radius: 5px;
            box-shadow: 0px 2px 5px #cccccc;
            padding: 10px;
            margin: 10px 0;
        }
        .footer {
            text-align: center;
            color: #777777;
            font-size: 14px;
            margin-top: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown('<div class="header">Hệ thống gợi ý sản phẩm</div>', unsafe_allow_html=True)

# cty: C:/Users/PC/Documents/NamNC/GUI_Recommender_System/
# nha: C:/Users/Windows 10/Downloads/Compressed/GUI_Recommender_System/

# Load your datasets
san_pham = pd.read_csv('San_pham.csv')
khach_hang = pd.read_csv('Khach_hang.csv')
danh_gia = pd.read_csv('Danh_gia.csv')

# Data preprocessing and merging
join_san_pham_danh_gia = pd.merge(danh_gia, san_pham, on='ma_san_pham', how='left')
df_ori = pd.merge(join_san_pham_danh_gia, khach_hang, on='ma_khach_hang', how='left')
dataframe = df_ori.copy()

df = dataframe[['ma_san_pham', 'ten_san_pham', 'mo_ta', 'so_sao']]

# function cần thiết

# Load cosine similarity matrix from the pickle file
with open('products_cosine_sim.pkl', 'rb') as f:
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

def analyze_month_statistics(df, selected_product):
    """Thống kê số lượng bình luận theo tháng và trực quan hóa trên biểu đồ."""
    st.write(f"**A. Số lượng bình luận theo tháng cho sản phẩm ID '{selected_product}':**")

    # Lọc dữ liệu cho sản phẩm cụ thể
    df = df[df['ma_san_pham'] == int(selected_product)]

    # Convert 'ngay_binh_luan' to datetime if it's not already
    df['ngay_binh_luan'] = pd.to_datetime(df['ngay_binh_luan'], dayfirst=True)

    # Nhóm dữ liệu theo tháng
    df['month'] = df['ngay_binh_luan'].dt.to_period('M')  # Creates a Period index for month
    month_count = df.groupby('month').size().reset_index(name='count')

    # Sort by 'month' column in descending order
    month_count = month_count.sort_values(by='month', ascending=False)

    # Lấy tháng có số lượt bình luận nhiều nhất
    if not month_count.empty:
        max_count = month_count['count'].max()
        top_month_row = month_count[month_count['count'] == max_count].iloc[0]
        st.write(f"Tháng có số lượt đánh giá nhiều nhất: {top_month_row['month']} với {top_month_row['count']} đánh giá.")

        # Trực quan hóa bằng Matplotlib
        plt.figure(figsize=(15, 7))
        
        # Vẽ biểu đồ cột
        bars = plt.bar(month_count['month'].astype(str), month_count['count'])
        
        # Tiêu đề và nhãn
        plt.title(f'Số lượng bình luận theo tháng - Sản phẩm {selected_product}', fontsize=15)
        plt.xlabel('Tháng', fontsize=12)
        plt.ylabel('Số lượng bình luận', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        # Thêm số lượng comment lên từng cột
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{int(height)}', 
                     ha='center', va='bottom', 
                     fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(plt)
    else:
        st.write("Không có bình luận nào.")

def analyze_comments_by_hour(df, product_id):
    """Thống kê số lượng bình luận theo khung giờ trong ngày cho một sản phẩm."""
    
    # Chọn ra bình luận của sản phẩm cụ thể
    product_comments = df[df['ma_san_pham'] == int(product_id)]
    
    # Chuyển đổi cột 'gio_binh_luan' sang kiểu datetime để lấy giờ
    product_comments['gio_binh_luan'] = product_comments['gio_binh_luan'].astype(str)
    product_comments.loc[:, 'hour'] = pd.to_datetime(product_comments['gio_binh_luan'], format='%H:%M').dt.hour

    # Đếm số lượng bình luận theo giờ
    hourly_counts = product_comments.groupby('hour').size().reset_index(name='count')

    # Hiển thị bảng thống kê
    st.write(f"**B. Số lượng bình luận theo giờ cho sản phẩm ID '{product_id}':**")
    #st.dataframe(hourly_counts)

    # Trực quan hóa bằng matplotlib
    plt.figure(figsize=(10, 5))
    bars = plt.bar(hourly_counts['hour'], hourly_counts['count'], color='skyblue')
    plt.xlabel('Khung giờ trong ngày')
    plt.ylabel('Số lượng bình luận')
    plt.title(f"Số lượng bình luận theo khung giờ cho sản phẩm ID {product_id}")
    plt.xticks(hourly_counts['hour'])  # Đảm bảo tất cả các giờ được hiển thị
    plt.grid(axis='y')

    # Thêm nhãn số liệu lên từng cột trong biểu đồ
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), va='bottom')  # va='bottom' để đặt nhãn ở trên cột

    # Hiển thị đồ thị trong Streamlit
    st.pyplot(plt)

def plot_star_ratings(danh_gia, user_input_int):
    # Chuyển đổi user_input_int sang kiểu int nếu cần
    user_input_int = int(user_input_int)
    
    # Thống kê số lượng đánh giá theo từng sao
    star_ratings_count = danh_gia[danh_gia['ma_san_pham'] == user_input_int]['so_sao'].value_counts().sort_index()
    
    # Đảm bảo có đủ các mức sao từ 1 đến 5
    full_star_ratings = pd.Series([0] * 5, index=range(1, 6))
    full_star_ratings.update(star_ratings_count)
    
    # Hiển thị thông tin trên Streamlit
    st.write(f"**IV. Số lượng đánh giá theo từng sao của sản phẩm vừa nhập:**")

    # Tạo biểu đồ dạng cột với matplotlib
    fig, ax = plt.subplots()
    ax.bar(full_star_ratings.index, full_star_ratings.values, color='skyblue')
    ax.set_title('Số lượng đánh giá theo từng sao')
    ax.set_xlabel('Sao')
    ax.set_ylabel('Số lượng đánh giá')
    ax.set_xticks(range(1, 6))

    # Thêm nhãn cho các cột
    for i, v in enumerate(full_star_ratings.values, start=1):
        ax.text(i, v, str(v), ha='center', va='bottom')

    # Trực quan hóa biểu đồ trong Streamlit
    st.pyplot(fig)


# Ham danh cho login
def get_customer_rated_products(customer_id):
    return danh_gia[danh_gia['ma_khach_hang'] == int(customer_id)].join(san_pham.set_index('ma_san_pham'), on='ma_san_pham', how='left')

def login_page():
    """Display the login page."""
    st.title("Login")
    ho_ten = st.text_input("Nhập họ tên:")
    ma_khach_hang = st.text_input("Nhập mã khách hàng:", type="password")

    if st.button("Đăng nhập"):
        if authenticate_user(ho_ten, int(ma_khach_hang)):
            st.session_state.user_logged_in = True
            st.session_state.customer_id = ma_khach_hang
            st.success("Đăng nhập thành công!")
            st.session_state.current_page = "user_products"  # Navigate to user products screen

        else:
            st.error("Thông tin đăng nhập không chính xác. Vui lòng thử lại.")
    

def authenticate_user(ho_ten, ma_khach_hang):
    """Authenticate user based on input."""
    user = khach_hang[(khach_hang['ho_ten'] == ho_ten) & (khach_hang['ma_khach_hang'] == ma_khach_hang)]
    return not user.empty

def show_user_products():
    """Display products rated by the logged-in customer."""
    customer_id = st.session_state.customer_id

    # Get the customer's full name from the customer data
    user_info = khach_hang[khach_hang['ma_khach_hang'] == int(customer_id)]
    
    if not user_info.empty:
        ho_ten = user_info.iloc[0]['ho_ten']
    else:
        ho_ten = "Người dùng"

    rated_products = get_customer_rated_products(customer_id)
    
    # Calculate the count of rated products
    total_rated_products = rated_products.shape[0]

    st.subheader(f"Chào mừng, {ho_ten}!!!")

    # Display total number of rated products
    st.write(f"### Bạn đã đánh giá tổng cộng {total_rated_products} sản phẩm.")

    if not rated_products.empty:
        st.write("### I. Sản phẩm mà bạn đã đánh giá:")
        st.dataframe(rated_products[['ten_san_pham', 'so_sao', 'mo_ta']])

        # Count ratings from 1 to 5 stars
        rating_counts = rated_products['so_sao'].value_counts().reindex(range(1, 6), fill_value=0)

        # Create a summary of ratings
        rating_summary = rated_products['so_sao'].value_counts().sort_index()

        # Create a bar chart with annotations
        st.write("### II. Thống kê đánh giá:")
        
        # Create a matplotlib figure for more customization
        fig, ax = plt.subplots()
        bars = ax.bar(rating_counts.index, rating_counts.values, color='skyblue')

        # Add data labels on top of the bars
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 0.2, int(yval), ha='center', va='bottom')

        ax.set_xlabel('Sao')
        ax.set_ylabel('Số lượng Đánh giá')
        ax.set_title('Số lượng Đánh giá theo Sao')

        st.pyplot(fig)  # Display the figure in Streamlit

        # Iterate through ratings 1 to 5 and display product details for each rating
        st.write("### III. Lịch sử đánh giá:")
        for rating in range(1, 6):
            rated_product_info = rated_products[rated_products['so_sao'] == rating]
            count = rating_summary.get(rating, 0)  # Get the count for each star rating

            if not rated_product_info.empty:
                # Merge with df_ori to get additional product information
                product_details = df_ori[df_ori['ma_san_pham'].isin(rated_product_info['ma_san_pham'])]

                # Remove duplicates based on 'ma_san_pham' to prevent showing duplicate products
                product_details_unique = product_details.drop_duplicates(subset=['ma_san_pham'])

                st.write(f"### Có {count} sản phẩm đã đánh giá {rating} sao:")
                st.dataframe(product_details_unique[['ten_san_pham', 'mo_ta','noi_dung_binh_luan','ngay_binh_luan','gio_binh_luan', 'gia_ban','gia_goc']])  # Adjust columns as needed
            else:
                st.write(f"### Không có sản phẩm nào bạn đã đánh giá {rating} sao.")

    else:
        st.write("Bạn chưa đánh giá sản phẩm nào.")


def logout():
    """Log out the user by clearing the session state."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.current_page = "login"  # Return to login page after logout
    st.success("Bạn đã đăng xuất thành công!")


# End ham danh cho login

# Common
st.title("Data Science Project")
st.write("##")

menu = ["Yêu cầu bài toán", "Xây dựng model", "Gợi ý cho người dùng", "Đăng nhập"]
choice = st.sidebar.selectbox('Menu', menu)
st.sidebar.write("""#### Thành viên thực hiện:""")
st.sidebar.write('1) CHẾ THỊ ÁNH TUYỀN')
st.sidebar.write('2) NGUYỄN CHẤN NAM')
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

    st.write("Thống Kê và Trực Quan Hóa Dữ Liệu Chăm Sóc Da Mặt")
    # st.write("##### 2. Visualize Ham and Spam")
    # fig1 = sns.countplot(data=data[['v1']], x='v1')    
    # st.pyplot(fig1.figure)

    # Đếm số lượng đánh giá cho từng sản phẩm
    danh_gia_count = join_san_pham_danh_gia.groupby(['ma_san_pham', 'ten_san_pham']).size().reset_index(name='so_luong_danh_gia')

    # Tìm sản phẩm có số lượng đánh giá nhiều nhất
    san_pham_nhieu_nhat = danh_gia_count.loc[danh_gia_count['so_luong_danh_gia'].idxmax()]

    # Tìm sản phẩm có số lượng đánh giá ít nhất
    san_pham_it_nhat = danh_gia_count.loc[danh_gia_count['so_luong_danh_gia'].idxmin()]

    # In ra thông tin sản phẩm với định dạng ba hàng
    st.write("##### Sản phẩm có số lượng đánh giá nhiều nhất:")
    st.write(f"Mã sản phẩm: {san_pham_nhieu_nhat['ma_san_pham']}")
    st.write(f"Tên sản phẩm: {san_pham_nhieu_nhat['ten_san_pham']}")
    st.write(f"Số lượng đánh giá: {san_pham_nhieu_nhat['so_luong_danh_gia']}")

    st.write("##### Sản phẩm có số lượng đánh giá ít nhất:")
    st.write(f"Mã sản phẩm: {san_pham_it_nhat['ma_san_pham']}")
    st.write(f"Tên sản phẩm: {san_pham_it_nhat['ten_san_pham']}")
    st.write(f"Số lượng đánh giá: {san_pham_it_nhat['so_luong_danh_gia']}")

    # 2. Trực quan hóa các loại thống kê
    st.header("Trực Quan Hóa Dữ Liệu")

    # Lấy dữ liệu cho phân bố đánh giá
    so_sao_counts = join_san_pham_danh_gia['so_sao'].value_counts().sort_index()

    # Phân bố điểm đánh giá
    st.subheader("Phân Bố Điểm Đánh Giá")
    plt.figure(figsize=(10, 6))

    # Vẽ biểu đồ phân bố
    sns.histplot(join_san_pham_danh_gia['so_sao'], bins=5, kde=True, color='blue')
    plt.title('Phân Bố Điểm Đánh Giá')
    plt.xlabel('Điểm Đánh Giá')
    plt.ylabel('Tần Suất')

    # Thêm số lượng cho mỗi loại đánh giá lên biểu đồ
    for index, value in so_sao_counts.items():
        plt.text(index, value, str(value), ha='center', va='bottom', fontsize=10)

    st.pyplot(plt)

    # Vẽ biểu đồ tròn thể hiện tỷ lệ giữa các loại đánh giá
    st.subheader("Tỷ Lệ Giữa Các Loại Đánh Giá")
    plt.figure(figsize=(8, 8))
    plt.pie(so_sao_counts, labels=so_sao_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("husl", len(so_sao_counts)))
    plt.title('Tỷ Lệ Giữa Các Loại Đánh Giá')
    plt.axis('equal')  # Để biểu đồ tròn không bị méo

    st.pyplot(plt)

    st.write("##### 3. Xây dựng model...")
    st.write("##### 4. Evaluation")
    # st.code("Score train:"+ str(round(score_train,2)) + " vs Score test:" + str(round(score_test,2)))
    # st.code("Accuracy:"+str(round(acc,2)))




    st.write("##### 5. Tổng kết: mô hình này chấp nhận được cho gợi ý đề xuất sản phẩm phù hợp với người dùng.")

elif choice == 'Gợi ý cho người dùng':

    # Streamlit UI
    st.image('hasaki_banner.jpg')
    st.title('Hệ thống gợi ý sản phẩm')

    # Display first 10 products
    st.subheader('Danh sách sản phẩm')
    san_pham['ma_san_pham'] = san_pham['ma_san_pham'].astype(str)
    st.dataframe(san_pham.head(10))  # Display the first 10 products
    

    user_input = st.text_input("Nhập tên sản phẩm, mã sản phẩm hoặc nội dung mô tả sản phẩm:")

    # Check if the user input is empty or only whitespace
    is_input_empty = user_input.strip() == ""

    # Hiển thị nút "Gợi ý", tắt nếu input trống
    if st.button('Gợi ý', disabled=is_input_empty):
        try:
            # Kiểm tra xem có phải là mã sản phẩm hay không
            user_input_int = int(user_input)

            # Lấy thông tin sản phẩm tương ứng
            product_info = df[df['ma_san_pham'] == user_input_int]
            
            if not product_info.empty:
                product_name = product_info['ten_san_pham'].values[0]
                product_desc = product_info['mo_ta'].values[0]
                product_rating = product_info['so_sao'].values[0]

                # Lấy gợi ý
                recommendations = get_recommendations_cosine(user_input_int)

                # Tạo hai tab: "Thông tin sản phẩm" và "Sản phẩm gợi ý"
                tabs = st.tabs(["Thông tin sản phẩm", "Sản phẩm gợi ý"])

                # Tab 1: Thông tin sản phẩm
                with tabs[0]:
                    st.markdown(f"<div class='big-font'>{product_name}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='desc-font'>{product_desc}</div>", unsafe_allow_html=True)
                    st.write(f"**I. Tên sản phẩm tương ứng mã sản phẩm vừa nhập:** {product_name}")
                    st.write(f"**II. Mô tả:** {product_desc}")
                    st.write(f"**III. Điểm trung bình:** {product_rating}")

                    # Hiển thị thống kê số lượng đánh giá theo từng sao
                    plot_star_ratings(danh_gia, user_input_int)

                    # Phân tích thống kê
                    filtered_df = df_ori[df_ori['ma_san_pham'] == user_input_int]
                    analyze_month_statistics(filtered_df, user_input_int)
                    analyze_comments_by_hour(filtered_df, user_input_int)

                # Tab 2: Sản phẩm gợi ý
                with tabs[1]:
                    if user_input:
                        # Lưu recommendations vào session state
                        st.session_state.recommendations = recommendations.copy()

                        # Chỉ lấy 5 sản phẩm gợi ý đầu tiên
                        top_recommendations = st.session_state.recommendations.head(5)

                        if not top_recommendations.empty:
                            st.write("**V. Top 5 sản phẩm gợi ý:**")

                            # Chuyển đổi kiểu cột 'ma_san_pham' thành kiểu string
                            top_recommendations['ma_san_pham'] = top_recommendations['ma_san_pham'].astype(str)

                            st.dataframe(top_recommendations[['ma_san_pham', 'ten_san_pham', 'mo_ta', 'so_sao']])

                            # Tạo các tab cho từng sản phẩm
                            tabs_recommendations = st.tabs([f"Sản phẩm {i + 1}" for i in range(len(top_recommendations))])

                            for i, product in enumerate(top_recommendations.iterrows()):
                                product_data = product[1]  # Lấy dữ liệu sản phẩm
                                with tabs_recommendations[i]:
                                    st.write(f"**Tên sản phẩm:** {product_data['ten_san_pham']}")
                                    st.write(f"**Mô tả:** {product_data['mo_ta']}")
                                    st.write(f"**Điểm trung bình:** {product_data['so_sao']}")

                                    product_id = int(product_data['ma_san_pham'])
                                    plot_star_ratings(danh_gia, product_id)  # Hiển thị biểu đồ đánh giá cho sản phẩm

                                    # Hiển thị thêm thông tin hoặc phân tích nếu cần
                                    filtered_df = df_ori[df_ori['ma_san_pham'] == product_id]
                                    analyze_month_statistics(filtered_df, product_id)
                                    analyze_comments_by_hour(filtered_df, product_id)
                        else:
                            st.write("Không tìm thấy sản phẩm gợi ý thỏa điều kiện.")

            else:
                st.warning('Không tìm thấy sản phẩm tương ứng với mã sản phẩm đã nhập!')

        except ValueError:
            # Nếu không phải mã sản phẩm số, xem như nhập vào là tên sản phẩm
            recommendations = get_recommendations_cosine(user_input)

            # Hiển thị chỉ tab "Sản phẩm gợi ý"
            tabs = st.tabs(["Sản phẩm gợi ý"])
            with tabs[0]:
                if recommendations.empty:
                    st.write("Không tìm thấy sản phẩm gợi ý thỏa điều kiện.")
                else:
                    st.write("**V. Top 5 sản phẩm gợi ý từ tên sản phẩm:**")
                    top_recommendations = recommendations.head(5)

                    # Chuyển đổi kiểu cột 'ma_san_pham' thành kiểu string
                    top_recommendations['ma_san_pham'] = top_recommendations['ma_san_pham'].astype(str)

                    st.dataframe(top_recommendations[['ma_san_pham', 'ten_san_pham', 'mo_ta', 'so_sao']])

                    # Tạo các tab cho từng sản phẩm gợi ý
                    tabs_recommendations = st.tabs([f"Sản phẩm {i + 1}" for i in range(len(top_recommendations))])

                    for i, product in enumerate(top_recommendations.iterrows()):
                        product_data = product[1]  # Lấy dữ liệu sản phẩm
                        with tabs_recommendations[i]:
                            st.write(f"**Tên sản phẩm:** {product_data['ten_san_pham']}")
                            st.write(f"**Mô tả:** {product_data['mo_ta']}")
                            st.write(f"**Điểm trung bình:** {product_data['so_sao']}")

                            product_id = product_data['ma_san_pham']
                            plot_star_ratings(danh_gia, product_id)  # Hiển thị biểu đồ đánh giá cho sản phẩm

                            # Hiển thị thêm thông tin hoặc phân tích nếu cần
                            filtered_df = df_ori[df_ori['ma_san_pham'] == product_id]
                            analyze_month_statistics(filtered_df, product_id)
                            analyze_comments_by_hour(filtered_df, product_id)


elif choice == 'Đăng nhập':
    # Kiểm tra xem người dùng đã đăng nhập hay chưa
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "login"

    if st.session_state.current_page == "login":
        login_page()
    elif st.session_state.current_page == "user_products":
        show_user_products()

        # Khởi tạo user_reviews nếu chưa có
        if 'user_reviews' not in st.session_state:
            st.session_state.user_reviews = []  # Hoặc một danh sách rỗng khác tùy theo cấu trúc bạn đang sử dụng

        # Nhận đầu vào từ người dùng
        login_user_input = st.text_input("Nhập tên sản phẩm, mã sản phẩm hoặc nội dung mô tả sản phẩm:")
        login_is_input_empty = login_user_input.strip() == ""

        # Hiển thị nút "Gợi ý", nếu đầu vào không trống
        if st.button('Gợi ý', disabled=login_is_input_empty):
            recommendations = None
            
            try:
                # Kiểm tra xem người dùng nhập vào có phải là mã sản phẩm không
                login_user_input_int = int(login_user_input)
                product_info = df[df['ma_san_pham'] == login_user_input_int]

                if not product_info.empty:
                    # Lấy thông tin sản phẩm
                    login_product_name = product_info['ten_san_pham'].values[0]
                    login_product_desc = product_info['mo_ta'].values[0]
                    login_product_rating = product_info['so_sao'].values[0]

                    # Tạo 2 tab: "Thông tin sản phẩm" và "Sản phẩm gợi ý"
                    tabs = st.tabs(["Thông tin sản phẩm", "Sản phẩm gợi ý"])

                    # Tab 1: Thông tin sản phẩm
                    with tabs[0]:
                        st.write(f"**Tên sản phẩm:** {login_product_name}")
                        st.write(f"**Mô tả:** {login_product_desc}")
                        st.write(f"**Điểm trung bình:** {login_product_rating}")

                        # Hiển thị biểu đồ đánh giá sao và phân tích
                        plot_star_ratings(danh_gia, login_user_input_int)
                        analyze_month_statistics(df_ori[df_ori['ma_san_pham'] == login_user_input_int], login_user_input_int)
                        analyze_comments_by_hour(df_ori[df_ori['ma_san_pham'] == login_user_input_int], login_user_input_int)

                    with tabs[1]:
                        st.write("**Top 5 sản phẩm gợi ý:**")
                        
                        # Lưu recommendations vào session state nếu chưa có
                        if 'recommendations' not in st.session_state:
                            st.session_state.recommendations = recommendations.copy()
                        else:
                            # Cập nhật recommendations trong session state
                            st.session_state.recommendations = recommendations.copy()

                        # Chọn 5 sản phẩm gợi ý đầu tiên
                        top_recommendations = st.session_state.recommendations.head(5)

                        # Chuyển đổi kiểu cột 'ma_san_pham' thành kiểu string
                        top_recommendations['ma_san_pham'] = top_recommendations['ma_san_pham'].astype(str)

                        # Hiển thị DataFrame
                        st.dataframe(top_recommendations)

                        # Tạo các tab con cho từng sản phẩm gợi ý
                        sub_tabs = st.tabs([f"Sản phẩm gợi ý {i+1}" for i in range(len(top_recommendations))])
                        
                        for i, product in enumerate(top_recommendations.iterrows()):
                            product_data = product[1]  # Lấy dữ liệu sản phẩm gợi ý
                            with sub_tabs[i]:
                                st.write(f"**Tên sản phẩm:** {product_data['ten_san_pham']}")
                                st.write(f"**Mô tả:** {product_data['mo_ta']}")
                                st.write(f"**Điểm trung bình:** {product_data['so_sao']}")

                                # Hiển thị biểu đồ hoặc phân tích khác nếu cần
                                product_id = int(product_data['ma_san_pham'])
                                plot_star_ratings(danh_gia, product_id)
                                
                                filtered_df = df_ori[df_ori['ma_san_pham'] == product_id]
                                analyze_month_statistics(filtered_df, product_id)
                                analyze_comments_by_hour(filtered_df, product_id)

                    # Lấy gợi ý sản phẩm
                    recommendations = get_recommendations_cosine(login_user_input_int)

            except ValueError:
                # Nếu không phải mã sản phẩm, coi như là nhập tên sản phẩm hoặc mô tả
                recommendations = get_recommendations_cosine(login_user_input)

            # Lọc sản phẩm gợi ý đã được đánh giá bởi người dùng
            reviewed_products = st.session_state.user_reviews
            if recommendations is not None:
                recommendations = recommendations[~recommendations['ma_san_pham'].isin(reviewed_products)]

                # Đoạn xử lý cho tab sản phẩm gợi ý
                if recommendations is not None and not recommendations.empty:
                    tabs = st.tabs(["Sản phẩm gợi ý"])
                    with tabs[0]:
                        st.write("**Top 5 sản phẩm gợi ý:**")
                        
                        # Lưu recommendations vào session state nếu chưa có
                        if 'recommendations' not in st.session_state:
                            st.session_state.recommendations = recommendations.copy()
                        else:
                            # Cập nhật recommendations trong session state
                            st.session_state.recommendations = recommendations.copy()

                        # Chọn 5 sản phẩm gợi ý đầu tiên
                        top_recommendations = st.session_state.recommendations.head(5)

                        # Chuyển đổi kiểu cột 'ma_san_pham' thành kiểu string
                        top_recommendations['ma_san_pham'] = top_recommendations['ma_san_pham'].astype(str)

                        # Hiển thị DataFrame
                        st.dataframe(top_recommendations)

                        # Tạo các tab con cho từng sản phẩm gợi ý
                        sub_tabs = st.tabs([f"Sản phẩm gợi ý {i+1}" for i in range(len(top_recommendations))])
                        
                        for i, product in enumerate(top_recommendations.iterrows()):
                            product_data = product[1]  # Lấy dữ liệu sản phẩm gợi ý
                            with sub_tabs[i]:
                                st.write(f"**Tên sản phẩm:** {product_data['ten_san_pham']}")
                                st.write(f"**Mô tả:** {product_data['mo_ta']}")
                                st.write(f"**Điểm trung bình:** {product_data['so_sao']}")

                                # Hiển thị biểu đồ hoặc phân tích khác nếu cần
                                product_id = int(product_data['ma_san_pham'])
                                plot_star_ratings(danh_gia, product_id)
                                
                                filtered_df = df_ori[df_ori['ma_san_pham'] == product_id]
                                analyze_month_statistics(filtered_df, product_id)
                                analyze_comments_by_hour(filtered_df, product_id)

            else:
                st.write("Không có sản phẩm gợi ý nào do bạn đã đánh giá tất cả hoặc không tìm thấy sản phẩm.")

        # Hiển thị nút "Đăng xuất"
        if st.button("Đăng xuất"):
            logout()