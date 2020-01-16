import React, { Component } from "react"
import {Avatar, Button} from "antd"
import ImageUnit from "./basasuya"


class ImageList extends Component {
    constructor(props) {
        super(props)

        // this.state = {
        //     imagelist: [
        //         {"path": "http://0.0.0.0:8000/1.png"},
        //         {"path": "http://0.0.0.0:8000/1.png"},
        //         {"path": "http://0.0.0.0:8000/1.png"},
        //         {"path": "http://0.0.0.0:8000/1.png"},
        //         {"path": "http://0.0.0.0:8000/1.png"},
        //         {"path": "http://0.0.0.0:8000/1.png"},
        //         {"path": "http://0.0.0.0:8000/1.png"},
        //         {"path": "http://0.0.0.0:8000/1.png"},
        //         {"path": "http://0.0.0.0:8000/1.png"},
        //         {"path": "http://0.0.0.0:8000/1.png"},
        //         {"path": "http://0.0.0.0:8000/1.png"},
        //         {"path": "http://0.0.0.0:8000/1.png"},

        //     ]
        // }
    }

    handleChooseComment (index) {
        if (this.props.onChange) {
            this.props.onChange(index)
        }
    }



    render() {
        const style = {
          maxHeight: this.props.totalHeight - 100
        };

        return <div className="imagelist" style={style}>
            {this.props.imagelist.map((key, i) => <ImageUnit path={key.path} key={i} index={i} onChooseComment={this.handleChooseComment.bind(this)}/>)}
            {/* {this.props.imagelist.map((key, i) =>  <Avatar src={key.path} shape="square" size={100} className="imageList" />)} */}
           
        </div>
    }
}

export default ImageList