import React, { Component } from "react"
import {Avatar, Button} from "antd"



class ImageUnit extends Component {
    constructor(props) {
        super(props)

    }


    handleClick() {
        if (this.props.onChooseComment) {
            this.props.onChooseComment(this.props.index)
        }
    }

    render() {
        // console.log(this.props.path)
        let path = this.props.path
        // console.log(this.props, path)
        return <Avatar src={path} shape="square" size={100} className="imageList" onClick={this.handleClick.bind(this)}/>
        // </div>
    }
}

export default ImageUnit