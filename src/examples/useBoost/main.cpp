#include <iostream>
#include <boost/asio.hpp>
#include <boost/array.hpp>
#include <boost/bind.hpp>
#include <vector>
#include <list>
#include <map>

using namespace std;

int main()
{
    cout << "hello testboost" << std::endl;
    //第一个阻塞响应Http请求
    // ASIO 主要IO类
    boost::asio::io_service ioService;
    //声明本机端口号(80),IP协议版本(IPV4)
    boost::asio::ip::tcp::endpoint e(boost::asio::ip::tcp::v4(), 80);
    //创建用来接收请求的接收器
    boost::asio::ip::tcp::acceptor acceptor(ioService, e);
    while (true)
    {
        // socket链接，用来记录请求过来的Socket信息，后期读取请求数据、响应数据都会用到
        boost::asio::ip::tcp::socket s(ioService);
        //开始阻塞等待请求
        acceptor.accept(s);
        //获取请求数据
        boost::array<char, 1024> buf;
        s.read_some(boost::asio::buffer(buf));
        //两种响应请求方式
        // boost::asio::write(s, boost::asio::buffer("HTTP/1.1 200 OK\r\nContent-Length: 13\r\n\r\nHello, world!"));
        s.write_some(boost::asio::buffer("HTTP/1.1 200 OK\r\nContent-Length: 13\r\n\r\nHello, world!"));
        cout << "接收到链接:" << s.remote_endpoint().address() << ":" << s.remote_endpoint().port() << " size:" << s.available() << endl;
        cout << "请求数据:" << buf.data();
        cout << "本地IP端口:" << s.local_endpoint().address() << ":" << s.local_endpoint().port() << endl;
    }
    return 0;
}