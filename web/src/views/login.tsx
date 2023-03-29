import React from 'react';
import { NavigateFunction, useNavigate } from 'react-router-dom';
import { withNavigate } from '../utils/react-utils';

import '../css/login.scss';
import Cookies from '../utils/Cookies';
import { fetchRequest } from '../utils/utils';

interface LoginInputAttrs {
  title: string;
  inType?: string;
  oninput: (
    event: React.ChangeEvent<HTMLInputElement>
  ) => void;
}

class InputGroup extends React.Component<LoginInputAttrs> {
  render() {
    return (
      <div className="input-group">
        <label>{ this.props.title }</label>
        <input 
          type={ this.props.inType || 'text' } 
          className="auth-in" 
          onInput={ this.props.oninput }
        />
      </div>
    );
  }
}

interface LoginRes {
  firstName: string;
  lastName: string;
  email: string;
  password: string;
  confirmPassword: string;
}

interface LoginAttrs {
  signup: boolean;
  navigate: NavigateFunction;
}

interface LoginState {
  valid: boolean;
  res: LoginRes;
}

class Login extends React.Component<LoginAttrs, LoginState> {
  constructor(props) {
    super(props);
    
    this.state = {
      valid: false,
      res: {
        firstName: '',
        lastName: '',
        email: '',
        password: '',
        confirmPassword: '',
      },
    }
  }

  authenticate(): void {
    fetchRequest<{uid: string}>(`/${this.props.signup ? 'createUser' : 'authenticateUser'}`, {
      method: 'POST',
      body: this.state.res,
    })
      .then(res => {
        if (res.uid) {
          console.log('success! ' + res.uid);
          Cookies.set('uid', res.uid, 2);
          this.props.navigate('/main');
        }
      })
      .catch(window.alert)
  }

  validate(): void {
    const res : LoginRes = this.state.res;

    if (this.props.signup)
      this.setState({
        valid: Boolean(
          res.firstName && res.lastName && res.email && 
          res.email.includes('@') && res.email.includes('.') &&
          res.password && res.confirmPassword && 
          res.password === res.confirmPassword
        )
      });
    else
      this.setState({
        valid: Boolean(res.email && res.password)
      });
  }

  handleChange(
    property: string, 
    event: React.ChangeEvent<HTMLInputElement>,
  ): void {
    this.setState(
      prevState => {
        let res = Object.assign({}, prevState.res);
        res[property] = event.target.value;
        return { res };
      }, 
      this.validate.bind(this)
    );
  }

  render() {
    return (
      <div className='auth-body'>
        <div className='auth-popup'>
          <div className='auth-content'>

            { this.props.signup &&
              <div className='hstack'>
                <InputGroup 
                  title='First Name'
                  oninput={ event => {
                    this.handleChange('firstName', event);
                  } }
                />
                <InputGroup 
                  title='Last Name'
                  oninput={ event => {
                    this.handleChange('lastName', event);
                  } }
                />
              </div>
            }

            <InputGroup 
              title='Email Address'
              oninput={ event => {
                this.handleChange('email', event);
              } }
            />
            <InputGroup 
              title='Password'
              inType='password'
              oninput={ event => {
                this.handleChange('password', event);
              } }
            />
            { this.props.signup &&
              <InputGroup 
                title='Confirm Password'
                inType='password'
                oninput={ event => {
                  this.handleChange('confirmPassword', event);
                } }
              />
            }
            <button
              className='login-button'
              onClick={ this.authenticate.bind(this) }
            >{
              this.props.signup ? 'Sign Up' : 'Log In'
            }</button>
            
          </div>
        </div>
      </div>
    );
  }
}

export default withNavigate(Login);